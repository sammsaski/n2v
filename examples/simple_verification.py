"""
Simple neural network verification example.

Demonstrates basic usage of NNV-Python for verifying a small feedforward network.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '..')

import nnv_py as nnv
from nnv_py.sets import Star, Box


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def example_basic_reachability():
    """Example 1: Basic reachability analysis."""
    print("=" * 60)
    print("Example 1: Basic Reachability Analysis")
    print("=" * 60)

    # Create a simple network
    model = SimpleNet()
    model.eval()

    # Manually set weights for reproducibility
    with torch.no_grad():
        model.fc1.weight = nn.Parameter(torch.tensor([
            [1.0, 0.5],
            [-0.5, 1.0],
            [0.2, -0.3],
            [-0.1, 0.4]
        ]))
        model.fc1.bias = nn.Parameter(torch.zeros(4))
        model.fc2.weight = nn.Parameter(torch.tensor([
            [0.5, -0.5, 0.3, 0.2],
            [-0.3, 0.4, -0.2, 0.5]
        ]))
        model.fc2.bias = nn.Parameter(torch.zeros(2))

    print(f"Network architecture: {model}")

    # Define input specification: perturbation around a point
    center = np.array([0.5, 0.5])
    epsilon = 0.1

    lb = center - epsilon
    ub = center + epsilon

    print(f"\nInput specification:")
    print(f"  Center: {center}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Bounds: [{lb[0]:.2f}, {ub[0]:.2f}] x [{lb[1]:.2f}, {ub[1]:.2f}]")

    # Create input set
    input_box = Box(lb.reshape(-1, 1), ub.reshape(-1, 1))
    print(f"\nInput set: {input_box}")

    # Wrap model for verification
    nn_verifier = nnv.NeuralNetwork(model, input_size=(2,))

    # Perform reachability analysis
    print("\nPerforming reachability analysis...")
    output_sets = nn_verifier.reach(input_box, method='approx-box')

    print(f"Number of output sets: {len(output_sets)}")
    for i, output_set in enumerate(output_sets):
        print(f"\nOutput set {i+1}: {output_set}")
        if isinstance(output_set, Box):
            print(f"  Output bounds:")
            print(f"    Dimension 1: [{output_set.lb[0, 0]:.4f}, {output_set.ub[0, 0]:.4f}]")
            print(f"    Dimension 2: [{output_set.lb[1, 0]:.4f}, {output_set.ub[1, 0]:.4f}]")

    return output_sets


def example_property_verification():
    """Example 2: Safety property verification."""
    print("\n" + "=" * 60)
    print("Example 2: Safety Property Verification")
    print("=" * 60)

    # Same network as before
    model = SimpleNet()
    model.eval()

    with torch.no_grad():
        model.fc1.weight = nn.Parameter(torch.tensor([
            [1.0, 0.5],
            [-0.5, 1.0],
            [0.2, -0.3],
            [-0.1, 0.4]
        ]))
        model.fc1.bias = nn.Parameter(torch.zeros(4))
        model.fc2.weight = nn.Parameter(torch.tensor([
            [0.5, -0.5, 0.3, 0.2],
            [-0.3, 0.4, -0.2, 0.5]
        ]))
        model.fc2.bias = nn.Parameter(torch.zeros(2))

    # Input specification
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    input_box = Box(lb.reshape(-1, 1), ub.reshape(-1, 1))

    print(f"Input specification: {input_box}")

    # Define safety property: output should satisfy y1 <= 2.0
    def safety_property(output_set):
        """Returns True if all outputs satisfy y1 <= 2.0"""
        if isinstance(output_set, Box):
            return output_set.ub[0, 0] <= 2.0
        elif hasattr(output_set, 'get_box'):
            box = output_set.get_box()
            return box.ub[0, 0] <= 2.0
        else:
            return False

    print("\nSafety property: output dimension 1 <= 2.0")

    # Verify property
    nn_verifier = nnv.NeuralNetwork(model)

    print("\nVerifying property...")
    is_safe = nn_verifier.verify_property(
        input_box,
        safety_property,
        method='approx-box'
    )

    print(f"\nVerification result: {'SAFE' if is_safe else 'UNSAFE (or UNKNOWN)'}")

    return is_safe


def example_set_operations():
    """Example 3: Set operations."""
    print("\n" + "=" * 60)
    print("Example 3: Set Operations")
    print("=" * 60)

    # Create two boxes
    box1 = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))
    box2 = Box(np.array([[0.5], [0.5]]), np.array([[1.5], [1.5]]))

    print(f"Box 1: {box1}")
    print(f"Box 2: {box2}")

    # Convert to zonotopes
    zono1 = box1.to_zono()
    zono2 = box2.to_zono()

    print(f"\nZono 1: {zono1}")
    print(f"Zono 2: {zono2}")

    # Minkowski sum
    zono_sum = zono1.minkowski_sum(zono2)
    print(f"\nMinkowski sum: {zono_sum}")
    box_sum = zono_sum.get_box()
    print(f"Bounding box of sum: [{box_sum.lb[0,0]:.2f}, {box_sum.ub[0,0]:.2f}] x "
          f"[{box_sum.lb[1,0]:.2f}, {box_sum.ub[1,0]:.2f}]")

    # Convex hull
    zono_hull = zono1.convex_hull(zono2)
    print(f"\nConvex hull: {zono_hull}")
    box_hull = zono_hull.get_box()
    print(f"Bounding box of hull: [{box_hull.lb[0,0]:.2f}, {box_hull.ub[0,0]:.2f}] x "
          f"[{box_hull.lb[1,0]:.2f}, {box_hull.ub[1,0]:.2f}]")

    # Affine transformation
    W = np.array([[2.0, 0.0], [0.0, 0.5]])
    b = np.array([[1.0], [0.5]])

    zono_transformed = zono1.affine_map(W, b)
    print(f"\nAffine transformation (W*x + b): {zono_transformed}")
    box_transformed = zono_transformed.get_box()
    print(f"Bounding box: [{box_transformed.lb[0,0]:.2f}, {box_transformed.ub[0,0]:.2f}] x "
          f"[{box_transformed.lb[1,0]:.2f}, {box_transformed.ub[1,0]:.2f}]")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NNV-Python: Neural Network Verification Examples")
    print("=" * 60 + "\n")

    # Run examples
    example_basic_reachability()
    example_property_verification()
    example_set_operations()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
