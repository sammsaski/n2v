"""
showcase_blackbox.py - Black-Box Model Verification

This script demonstrates probabilistic verification on models where
deterministic methods CANNOT be applied:
1. External API models (no architecture access)
2. Non-differentiable models
3. Ensemble models
4. Models with unsupported layers

This is the UNIQUE strength of probabilistic verification.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os

from n2v.probabilistic import verify
from n2v.sets import Box


# Output directory for visualizations
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("PROBABILISTIC VERIFICATION: BLACK-BOX MODELS")
    print("=" * 70)

    print("""
Probabilistic verification treats the model as a BLACK BOX.
It only needs a callable that maps inputs to outputs.

This enables verification of models that deterministic methods
cannot handle at all:
- External APIs (no architecture access)
- Ensemble models (multiple models combined)
- Models with unsupported layers
- Non-differentiable operations
""")

    np.random.seed(42)
    torch.manual_seed(42)

    # =========================================================================
    # Scenario 1: Ensemble Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 1: ENSEMBLE MODEL")
    print("=" * 70)

    print("""
An ensemble combines multiple models' predictions.
Deterministic reachability cannot handle this composition,
but probabilistic verification works seamlessly.
""")

    # Create three different models
    model1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    model2 = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))
    model3 = nn.Sequential(nn.Linear(4, 12), nn.Sigmoid(), nn.Linear(12, 2))

    for m in [model1, model2, model3]:
        m.eval()

    def ensemble_fn(x):
        """Ensemble: average of three models."""
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y1 = model1(x_tensor).numpy()
            y2 = model2(x_tensor).numpy()
            y3 = model3(x_tensor).numpy()
            return (y1 + y2 + y3) / 3

    print("Ensemble architecture:")
    print("  Model 1: Linear(4,8) -> ReLU -> Linear(8,2)")
    print("  Model 2: Linear(4,16) -> Tanh -> Linear(16,2)")
    print("  Model 3: Linear(4,12) -> Sigmoid -> Linear(12,2)")
    print("  Ensemble: Average of all three")

    # Input region
    lb = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    ub = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_box = Box(lb, ub)

    print(f"\nInput region: [-1, 1]^4")

    print("\n--- Running Probabilistic Verification ---")
    start = time.time()
    result = verify(
        model=ensemble_fn,
        input_set=input_box,
        m=1000,
        epsilon=0.05,
        surrogate='clipping_block',
        training_samples=500,
        seed=42,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Output bounds:")
    print(f"    Dim 0: [{result.lb.flatten()[0]:.4f}, {result.ub.flatten()[0]:.4f}]")
    print(f"    Dim 1: [{result.lb.flatten()[1]:.4f}, {result.ub.flatten()[1]:.4f}]")
    print(f"  {result.get_guarantee_string()}")

    # Validate
    n_test = 5000
    test_inputs = np.random.uniform(lb, ub, size=(n_test, 4)).astype(np.float32)
    test_outputs = ensemble_fn(test_inputs)
    inside = np.all(
        (test_outputs >= result.lb.flatten()) &
        (test_outputs <= result.ub.flatten()),
        axis=1
    )
    print(f"\n  Empirical coverage ({n_test} samples): {np.mean(inside):.4f}")

    # =========================================================================
    # Scenario 2: Model with Unsupported Layers
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 2: MODEL WITH UNSUPPORTED LAYERS")
    print("=" * 70)

    print("""
n2v's deterministic methods support specific layer types.
Models with Softmax, LayerNorm, or custom layers cannot be verified.
Probabilistic verification handles ANY layer.
""")

    # Create model with unsupported layers
    class ModelWithSoftmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 16)
            self.norm = nn.LayerNorm(16)  # Not supported in deterministic
            self.fc2 = nn.Linear(16, 3)
            self.softmax = nn.Softmax(dim=-1)  # Not supported

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.norm(x)
            x = self.fc2(x)
            return self.softmax(x)

    softmax_model = ModelWithSoftmax()
    softmax_model.eval()

    def softmax_fn(x):
        with torch.no_grad():
            return softmax_model(torch.tensor(x, dtype=torch.float32)).numpy()

    print("Model architecture:")
    print("  Linear(4, 16) -> ReLU -> LayerNorm(16) -> Linear(16, 3) -> Softmax")
    print("\n  LayerNorm and Softmax are NOT supported in deterministic methods!")

    print("\n--- Running Probabilistic Verification ---")
    start = time.time()
    result = verify(
        model=softmax_fn,
        input_set=input_box,
        m=1000,
        epsilon=0.05,
        surrogate='clipping_block',
        training_samples=500,
        seed=42,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Output bounds (Softmax output, sums to 1):")
    for i in range(3):
        print(f"    Class {i}: [{result.lb.flatten()[i]:.4f}, {result.ub.flatten()[i]:.4f}]")
    print(f"  {result.get_guarantee_string()}")

    # Check softmax property (outputs sum to 1)
    test_out = softmax_fn(np.random.uniform(lb, ub, size=(100, 4)).astype(np.float32))
    print(f"\n  Softmax property check: outputs sum to {np.mean(np.sum(test_out, axis=1)):.4f}")

    # =========================================================================
    # Scenario 3: External API (Simulated)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 3: EXTERNAL API (SIMULATED)")
    print("=" * 70)

    print("""
When using an external ML API (like OpenAI, AWS, etc.),
you have NO access to the model architecture.
Probabilistic verification is the ONLY option.
""")

    # Simulate an external API
    class ExternalAPI:
        """Simulates an external ML API."""
        def __init__(self):
            # Hidden model that user cannot access
            self._model = nn.Sequential(
                nn.Linear(5, 32),
                nn.GELU(),  # Modern activation
                nn.Dropout(0.1),
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, 2)
            )
            self._model.eval()
            self.call_count = 0

        def predict(self, x: np.ndarray) -> np.ndarray:
            """API endpoint - returns predictions."""
            self.call_count += len(x)
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32)
                return self._model(x_tensor).numpy()

    api = ExternalAPI()

    print("External API:")
    print("  - Architecture: UNKNOWN (proprietary)")
    print("  - Interface: predict(x: np.ndarray) -> np.ndarray")
    print("  - Input: 5 dimensions")
    print("  - Output: 2 dimensions")

    # Define input region
    lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_box = Box(lb, ub)

    print(f"\nInput region: [0, 1]^5")

    print("\n--- Running Probabilistic Verification ---")
    api.call_count = 0
    start = time.time()
    result = verify(
        model=api.predict,
        input_set=input_box,
        m=500,
        epsilon=0.05,
        surrogate='clipping_block',
        training_samples=250,
        seed=42,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  API calls made: {api.call_count}")
    print(f"  Output bounds:")
    print(f"    Dim 0: [{result.lb.flatten()[0]:.4f}, {result.ub.flatten()[0]:.4f}]")
    print(f"    Dim 1: [{result.lb.flatten()[1]:.4f}, {result.ub.flatten()[1]:.4f}]")
    print(f"  {result.get_guarantee_string()}")

    # =========================================================================
    # Scenario 4: Non-Differentiable Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 4: NON-DIFFERENTIABLE MODEL")
    print("=" * 70)

    print("""
Some models include non-differentiable operations like:
- Argmax / Top-k selection
- Discrete decisions (if-else)
- Quantization

Deterministic methods require differentiability for relaxations.
Probabilistic verification doesn't care!
""")

    # Model with argmax-based selection
    class NonDiffModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch1 = nn.Linear(4, 8)
            self.branch2 = nn.Linear(4, 8)
            self.selector = nn.Linear(4, 2)  # Decides which branch
            self.output = nn.Linear(8, 2)

        def forward(self, x):
            # Non-differentiable selection
            scores = self.selector(x)
            choice = torch.argmax(scores, dim=-1, keepdim=True)

            b1 = torch.relu(self.branch1(x))
            b2 = torch.relu(self.branch2(x))

            # Use choice to select branch (non-differentiable)
            selected = torch.where(choice == 0, b1, b2)
            return self.output(selected)

    nondiff_model = NonDiffModel()
    nondiff_model.eval()

    def nondiff_fn(x):
        with torch.no_grad():
            return nondiff_model(torch.tensor(x, dtype=torch.float32)).numpy()

    print("Model architecture:")
    print("  Selector: Linear(4,2) -> Argmax (non-differentiable!)")
    print("  Branch 1: Linear(4,8) -> ReLU")
    print("  Branch 2: Linear(4,8) -> ReLU")
    print("  Output: Selected branch -> Linear(8,2)")

    lb = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    ub = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_box = Box(lb, ub)

    print("\n--- Running Probabilistic Verification ---")
    start = time.time()
    result = verify(
        model=nondiff_fn,
        input_set=input_box,
        m=1000,
        epsilon=0.05,
        surrogate='clipping_block',
        training_samples=500,
        seed=42,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Output bounds:")
    print(f"    Dim 0: [{result.lb.flatten()[0]:.4f}, {result.ub.flatten()[0]:.4f}]")
    print(f"    Dim 1: [{result.lb.flatten()[1]:.4f}, {result.ub.flatten()[1]:.4f}]")
    print(f"  {result.get_guarantee_string()}")

    # =========================================================================
    # Generate Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        scenarios = [
            ('Ensemble Model', ensemble_fn, Box(np.array([-1]*4, dtype=np.float32),
                                                np.array([1]*4, dtype=np.float32))),
            ('Softmax Output', softmax_fn, Box(np.array([-1]*4, dtype=np.float32),
                                               np.array([1]*4, dtype=np.float32))),
            ('External API', api.predict, Box(np.array([0]*5, dtype=np.float32),
                                              np.array([1]*5, dtype=np.float32))),
            ('Non-Diff Model', nondiff_fn, Box(np.array([-1]*4, dtype=np.float32),
                                               np.array([1]*4, dtype=np.float32)))
        ]

        for idx, (title, model_fn, ibox) in enumerate(scenarios):
            ax = axes[idx // 2, idx % 2]

            # Get result
            res = verify(
                model=model_fn,
                input_set=ibox,
                m=500,
                epsilon=0.05,
                surrogate='naive',
                seed=42,
                verbose=False
            )

            # Sample outputs
            n_samples = 500
            lb_flat = ibox.lb.flatten()
            ub_flat = ibox.ub.flatten()
            samples = np.random.uniform(lb_flat, ub_flat, size=(n_samples, ibox.dim)).astype(np.float32)
            outputs = model_fn(samples)

            # Plot first two dimensions
            ax.scatter(outputs[:, 0], outputs[:, 1], alpha=0.3, s=10, c='blue', label='Samples')

            # Plot bounds (filled rectangle)
            rect = plt.Rectangle(
                (res.lb.flatten()[0], res.lb.flatten()[1]),
                res.ub.flatten()[0] - res.lb.flatten()[0],
                res.ub.flatten()[1] - res.lb.flatten()[1],
                fill=True, facecolor='#e74c3c', alpha=0.3,
                edgecolor='#e74c3c', linewidth=2, label='95% Coverage Bounds'
            )
            ax.add_patch(rect)

            ax.set_xlabel('Output Dimension 0')
            ax.set_ylabel('Output Dimension 1')
            ax.set_title(f'{title}\n(Black-box verified)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Expand axis limits to show reach set more clearly
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min
            ax.set_xlim(x_min - 0.5 * x_range, x_max + 0.5 * x_range)
            ax.set_ylim(y_min - 0.5 * y_range, y_max + 0.5 * y_range)

        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, 'blackbox_showcase.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available - skipping visualization")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
PROBABILISTIC VERIFICATION FOR BLACK-BOX MODELS:

UNIQUE CAPABILITIES:
1. Ensemble Models - Combines multiple models' predictions
2. Unsupported Layers - LayerNorm, Softmax, custom layers
3. External APIs - No architecture access required
4. Non-Differentiable Ops - Argmax, discrete decisions

REQUIREMENTS:
- Only needs a callable: np.ndarray -> np.ndarray
- No architecture knowledge needed
- No differentiability required

LIMITATIONS:
- NOT sound (coverage guarantee only)
- Bounds may be conservative
- Needs many forward passes (m samples)

USE CASES:
- ML-as-a-Service APIs (OpenAI, AWS, etc.)
- Complex model pipelines
- Research prototyping with new architectures
- Models with proprietary components
""")


if __name__ == "__main__":
    main()
