"""
Neural network verification module.

Provides PyTorch model wrappers for reachability analysis.

The primary API is the NeuralNetwork class, which wraps PyTorch models
and provides the reach() method for reachability analysis.
"""

from n2v.nn.neural_network import NeuralNetwork
from n2v.nn.reach import ReachConfig

try:
    from n2v.nn.spiking_neural_network import SpikingNeuralNetwork, SNNReachConfig
    _SNN_AVAILABLE = True
except ImportError:
    _SNN_AVAILABLE = False

__all__ = ["NeuralNetwork", "ReachConfig"]
if _SNN_AVAILABLE:
    __all__ += ["SpikingNeuralNetwork", "SNNReachConfig"]
