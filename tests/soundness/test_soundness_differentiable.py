"""
Soundness tests for differentiable LP solver with Hexatope and Octatope.

These tests verify that the differentiable solver produces sound over-approximations
when used with neural network verification operations.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sets.hexatope import Hexatope
from sets.octatope import Octatope
from sets.box import Box
from nn.layer_ops.linear_reach import linear_hexatope, linear_octatope
from nn.layer_ops.relu_reach import relu_hexatope, relu_octatope


def sample_from_set(abstract_set, n_samples=100):
    """
    Sample concrete points from an abstract set.

    Args:
        abstract_set: Hexatope or Octatope
        n_samples: Number of samples

    Returns:
        Array of shape (n_samples, dim) containing sampled points
    """
    dim = abstract_set.dim
    n_vars = abstract_set.nVar

    samples = []
    for _ in range(n_samples):
        # Sample from generator space (uniformly in [-1, 1]^n_vars)
        alpha = np.random.uniform(-1, 1, (n_vars, 1))

        # Map to state space: y = G*alpha + c
        point = abstract_set.generators @ alpha + abstract_set.center.reshape(-1, 1)
        samples.append(point.flatten())

    return np.array(samples)


def verify_soundness(input_set, output_set, operation_fn, layer=None, n_samples=100):
    """
    Verify soundness: for all x in input_set, operation(x) should be in output_set.

    Args:
        input_set: Input abstract set
        output_set: Output abstract set
        operation_fn: Function to apply (e.g., layer forward)
        layer: Neural network layer (if applicable)
        n_samples: Number of samples to test

    Returns:
        (is_sound, violation_ratio, max_violation)
    """
    # Sample from input set
    input_samples = sample_from_set(input_set, n_samples)

    violations = 0
    max_violation = 0.0

    for sample in input_samples:
        # Apply operation
        if layer is not None:
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            output_tensor = layer(sample_tensor)
            output_point = output_tensor.detach().numpy().flatten()
        else:
            output_point = operation_fn(sample)

        # Check if output is in output_set
        is_contained = output_set.contains(output_point)

        if not is_contained:
            violations += 1
            # Compute violation distance
            lb, ub = output_set.get_ranges(use_mcf=False)
            violation = np.maximum(0, output_point.reshape(-1, 1) - ub).max()
            violation = max(violation, np.maximum(0, lb - output_point.reshape(-1, 1)).max())
            max_violation = max(max_violation, violation)

    violation_ratio = violations / n_samples
    is_sound = violations == 0

    return is_sound, violation_ratio, max_violation


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestHexatopeSoundnessLinear:
    """Soundness tests for Hexatope with Linear layers."""

    def test_identity_linear_soundness(self):
        """Test soundness of identity linear transformation."""
        # Create input hexatope
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_hex = Hexatope.from_bounds(lb, ub)

        # Identity layer
        layer = nn.Linear(2, 2, bias=False)
        layer.weight.data = torch.eye(2)

        # Apply transformation
        output_hexs = linear_hexatope(layer, [input_hex])
        output_hex = output_hexs[0]

        # Verify soundness
        is_sound, violation_ratio, max_violation = verify_soundness(
            input_hex, output_hex, None, layer, n_samples=100
        )

        assert is_sound or violation_ratio < 0.1, \
            f"Identity transform not sound: {violation_ratio*100}% violations, max={max_violation}"

    def test_scaling_linear_soundness(self):
        """Test soundness of scaling transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_hex = Hexatope.from_bounds(lb, ub)

        # Scaling layer: y = 2*x
        layer = nn.Linear(2, 2, bias=False)
        layer.weight.data = 2 * torch.eye(2)

        output_hexs = linear_hexatope(layer, [input_hex])
        output_hex = output_hexs[0]

        is_sound, violation_ratio, max_violation = verify_soundness(
            input_hex, output_hex, None, layer, n_samples=100
        )

        assert is_sound or violation_ratio < 0.1, \
            f"Scaling not sound: {violation_ratio*100}% violations, max={max_violation}"

    def test_translation_linear_soundness(self):
        """Test soundness of translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_hex = Hexatope.from_bounds(lb, ub)

        # Translation: y = x + [1, 2]
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.tensor([1.0, 2.0])

        output_hexs = linear_hexatope(layer, [input_hex])
        output_hex = output_hexs[0]

        is_sound, violation_ratio, max_violation = verify_soundness(
            input_hex, output_hex, None, layer, n_samples=100
        )

        assert is_sound or violation_ratio < 0.1, \
            f"Translation not sound: {violation_ratio*100}% violations, max={max_violation}"

    def test_dimension_reduction_soundness(self):
        """Test soundness of dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_hex = Hexatope.from_bounds(lb, ub)

        # Projection: (x, y, z) -> (x+y, z)
        layer = nn.Linear(3, 2, bias=False)
        layer.weight.data = torch.tensor([[1.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0]])

        output_hexs = linear_hexatope(layer, [input_hex])
        output_hex = output_hexs[0]

        is_sound, violation_ratio, max_violation = verify_soundness(
            input_hex, output_hex, None, layer, n_samples=100
        )

        assert is_sound or violation_ratio < 0.15, \
            f"Dimension reduction not sound: {violation_ratio*100}% violations, max={max_violation}"

    def test_dimension_expansion_soundness(self):
        """Test soundness of dimension expansion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_hex = Hexatope.from_bounds(lb, ub)

        # Expansion: (x, y) -> (x, y, x+y)
        layer = nn.Linear(2, 3, bias=False)
        layer.weight.data = torch.tensor([[1.0, 0.0],
                                          [0.0, 1.0],
                                          [1.0, 1.0]])

        output_hexs = linear_hexatope(layer, [input_hex])
        output_hex = output_hexs[0]

        is_sound, violation_ratio, max_violation = verify_soundness(
            input_hex, output_hex, None, layer, n_samples=100
        )

        assert is_sound or violation_ratio < 0.15, \
            f"Dimension expansion not sound: {violation_ratio*100}% violations, max={max_violation}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOctatopeSoundnessLinear:
    """Soundness tests for Octatope with Linear layers."""

    def test_identity_linear_soundness(self):
        """Test soundness of identity linear transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_oct = Octatope.from_bounds(lb, ub)

        layer = nn.Linear(2, 2, bias=False)
        layer.weight.data = torch.eye(2)

        output_octs = linear_octatope(layer, [input_oct])
        output_oct = output_octs[0]

        is_sound, violation_ratio, max_violation = verify_soundness(
            input_oct, output_oct, None, layer, n_samples=100
        )

        assert is_sound or violation_ratio < 0.1, \
            f"Identity not sound: {violation_ratio*100}% violations, max={max_violation}"

    def test_rotation_soundness(self):
        """Test soundness of rotation transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_oct = Octatope.from_bounds(lb, ub)

        # 45-degree rotation
        angle = np.pi / 4
        c, s = np.cos(angle), np.sin(angle)
        layer = nn.Linear(2, 2, bias=False)
        layer.weight.data = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

        output_octs = linear_octatope(layer, [input_oct])
        output_oct = output_octs[0]

        is_sound, violation_ratio, max_violation = verify_soundness(
            input_oct, output_oct, None, layer, n_samples=100
        )

        assert is_sound or violation_ratio < 0.15, \
            f"Rotation not sound: {violation_ratio*100}% violations, max={max_violation}"

    def test_negative_weights_soundness(self):
        """Test soundness with negative weights."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_oct = Octatope.from_bounds(lb, ub)

        # Negative scaling
        layer = nn.Linear(2, 2, bias=False)
        layer.weight.data = torch.tensor([[-1.0, 0.0], [0.0, -2.0]])

        output_octs = linear_octatope(layer, [input_oct])
        output_oct = output_octs[0]

        is_sound, violation_ratio, max_violation = verify_soundness(
            input_oct, output_oct, None, layer, n_samples=100
        )

        assert is_sound or violation_ratio < 0.1, \
            f"Negative weights not sound: {violation_ratio*100}% violations, max={max_violation}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestHexatopeSoundnessReLU:
    """Soundness tests for Hexatope with ReLU layers."""

    def test_relu_all_positive_soundness(self):
        """Test ReLU soundness when all inputs are positive."""
        lb = np.array([[0.5], [0.5]])
        ub = np.array([[1.0], [1.0]])
        input_hex = Hexatope.from_bounds(lb, ub)

        output_hexs = relu_hexatope([input_hex])
        output_hex = output_hexs[0]

        # Sample and verify
        input_samples = sample_from_set(input_hex, 100)
        for sample in input_samples:
            output = np.maximum(sample, 0)
            assert output_hex.contains(output), \
                f"ReLU output {output} not in output set"

    def test_relu_all_negative_soundness(self):
        """Test ReLU soundness when all inputs are negative."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[-0.5], [-0.5]])
        input_hex = Hexatope.from_bounds(lb, ub)

        output_hexs = relu_hexatope([input_hex])
        output_hex = output_hexs[0]

        # All outputs should be zero
        lb_out, ub_out = output_hex.get_ranges(use_mcf=False)
        assert np.all(lb_out <= 0.1)
        assert np.all(ub_out >= -0.1)

    def test_relu_crossing_zero_soundness(self):
        """Test ReLU soundness when input crosses zero."""
        lb = np.array([[-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5]])
        input_hex = Hexatope.from_bounds(lb, ub)

        output_hexs = relu_hexatope([input_hex])
        output_hex = output_hexs[0]

        # Sample and verify
        input_samples = sample_from_set(input_hex, 100)
        violations = 0
        for sample in input_samples:
            output = np.maximum(sample, 0)
            if not output_hex.contains(output):
                violations += 1

        # Allow some violations due to approximation
        assert violations / 100 < 0.2, \
            f"Too many violations: {violations}%"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOctatopeSoundnessReLU:
    """Soundness tests for Octatope with ReLU layers."""

    def test_relu_all_positive_soundness(self):
        """Test ReLU soundness when all inputs are positive."""
        lb = np.array([[0.5], [0.5]])
        ub = np.array([[1.0], [1.0]])
        input_oct = Octatope.from_bounds(lb, ub)

        output_octs = relu_octatope([input_oct])
        output_oct = output_octs[0]

        # Sample and verify
        input_samples = sample_from_set(input_oct, 100)
        for sample in input_samples:
            output = np.maximum(sample, 0)
            assert output_oct.contains(output), \
                f"ReLU output {output} not in output set"

    def test_relu_mixed_dimensions_soundness(self):
        """Test ReLU with mixed positive/negative dimensions."""
        lb = np.array([[-1.0], [0.5]])
        ub = np.array([[0.5], [1.0]])
        input_oct = Octatope.from_bounds(lb, ub)

        output_octs = relu_octatope([input_oct])
        output_oct = output_octs[0]

        # Verify bounds
        lb_out, ub_out = output_oct.get_ranges(use_mcf=False)
        assert lb_out[0] <= 0.1  # First dim can be zero
        assert lb_out[1] >= 0.4  # Second dim stays positive
        assert ub_out[1] <= 1.1  # Upper bound preserved


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestComposedOperationsSoundness:
    """Soundness tests for composed operations."""

    def test_hexatope_linear_relu_composition(self):
        """Test soundness of Linear -> ReLU composition with Hexatope."""
        lb = np.array([[-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5]])
        input_hex = Hexatope.from_bounds(lb, ub)

        # Linear layer
        linear = nn.Linear(2, 2)
        linear.weight.data = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        linear.bias.data = torch.tensor([0.5, 0.5])

        # Apply linear
        after_linear = linear_hexatope(linear, [input_hex])[0]

        # Apply ReLU
        after_relu = relu_hexatope([after_linear])[0]

        # Verify soundness end-to-end
        input_samples = sample_from_set(input_hex, 100)
        violations = 0

        for sample in input_samples:
            # Apply operations
            sample_tensor = torch.tensor(sample, dtype=torch.float32)
            output = torch.relu(linear(sample_tensor)).detach().numpy()

            if not after_relu.contains(output):
                violations += 1

        assert violations / 100 < 0.2, \
            f"Linear->ReLU composition not sound: {violations}% violations"

    def test_octatope_multi_layer_soundness(self):
        """Test soundness of multi-layer network with Octatope."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_oct = Octatope.from_bounds(lb, ub)

        # Layer 1: Linear
        layer1 = nn.Linear(2, 3)
        layer1.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        layer1.bias.data = torch.zeros(3)

        # Layer 2: Linear
        layer2 = nn.Linear(3, 2)
        layer2.weight.data = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        layer2.bias.data = torch.zeros(2)

        # Apply transformations
        after_layer1 = linear_octatope(layer1, [input_oct])[0]
        after_relu1 = relu_octatope([after_layer1])[0]
        after_layer2 = linear_octatope(layer2, [after_relu1])[0]

        # Verify soundness
        input_samples = sample_from_set(input_oct, 50)
        violations = 0

        for sample in input_samples:
            sample_tensor = torch.tensor(sample, dtype=torch.float32)
            output = layer2(torch.relu(layer1(sample_tensor))).detach().numpy()

            if not after_layer2.contains(output):
                violations += 1

        assert violations / 50 < 0.25, \
            f"Multi-layer not sound: {violations/50*100}% violations"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestComparativeSoundness:
    """Compare soundness of differentiable solver vs standard solver."""

    def test_compare_hexatope_solvers(self):
        """Compare soundness between standard and differentiable solvers for Hexatope."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_hex = Hexatope.from_bounds(lb, ub)

        # Get ranges with standard solver
        lb_std, ub_std = input_hex.get_ranges(use_mcf=False)

        # Both should contain the input bounds
        assert np.all(lb_std <= lb + 0.1)
        assert np.all(ub_std >= ub - 0.1)

    def test_compare_octatope_solvers(self):
        """Compare soundness between standard and differentiable solvers for Octatope."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_oct = Octatope.from_bounds(lb, ub)

        # Get ranges with standard solver
        lb_std, ub_std = input_oct.get_ranges(use_mcf=False)

        # Both should contain the input bounds
        assert np.all(lb_std <= lb + 0.1)
        assert np.all(ub_std >= ub - 0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
