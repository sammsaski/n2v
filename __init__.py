"""
NNV-Python: Neural Network Verification Tool for Python/PyTorch

A formal verification tool for deep learning models, supporting reachability
analysis and robustness verification using set-based methods.

Translated from the original MATLAB NNV tool by the VeriVital research group.
"""

__version__ = "2.0.0"
__author__ = "NNV Team"

from nnv_py.sets import Star, Zono, Box, ImageStar, ImageZono
from nnv_py.nn import NeuralNetwork
from nnv_py import utils

__all__ = [
    "Star",
    "Zono",
    "Box",
    "ImageStar",
    "ImageZono",
    "NeuralNetwork",
    "utils",
]
