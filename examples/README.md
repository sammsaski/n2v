# n2v Examples

Examples demonstrating neural network verification with n2v.

## Getting Started

```bash
pip install -e .  # from the n2v root directory

# Simplest example
python examples/simple_verification.py

# MNIST tutorial (run in order)
cd examples/Tutorial
python train_fc.py
python verify_fc.py
```

## Examples

| Directory | Description |
|-----------|-------------|
| [simple_verification.py](simple_verification.py) | Basic feedforward verification with Star sets |
| [Tutorial/](Tutorial/) | Train and verify MNIST classifiers (FC and CNN) |
| [ACASXu/](ACASXu/) | ACAS Xu benchmark (186 instances, VNN-COMP format) |
| [VNN-COMP/](VNN-COMP/) | VNN-COMP 2025 infrastructure (28 benchmarks) |
| [ProbVer/](ProbVer/) | Probabilistic verification via conformal inference |
| [Octatope/](Octatope/) | Hexatope/Octatope set comparisons |
| [CompareNNV/](CompareNNV/) | Comparison with MATLAB NNV |
| [CompareReachability/](CompareReachability/) | Reachability method comparisons |

## Recommended Order

1. **simple_verification.py** -- Understand Star sets and basic reachability
2. **Tutorial/** -- Train and verify MNIST models (FC then CNN)
3. **ACASXu/** -- Real-world benchmark with falsification + verification
4. **ProbVer/** -- Probabilistic verification for large models
