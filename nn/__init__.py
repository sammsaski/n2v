"""
Neural network verification module.

Provides PyTorch model wrappers for reachability analysis.

The primary API is the NeuralNetwork class, which wraps PyTorch models
and provides the reach() method for reachability analysis.
"""

from n2v.nn.neural_network import NeuralNetwork

__all__ = ["NeuralNetwork"]
