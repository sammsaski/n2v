"""
n2v: Neural Network Verification Tool for Python/PyTorch

A formal verification tool for deep learning models, supporting reachability
analysis and robustness verification using set-based methods.

Translated from the original MATLAB NNV tool by the VeriVital research group.
"""

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("n2v")
except PackageNotFoundError:
    __version__ = "0.1.1"  # fallback for editable/dev installs
__author__ = "NNV Team"

from n2v.sets import Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope, HalfSpace, ProbabilisticBox
from n2v.nn import NeuralNetwork
from n2v import utils
from n2v import probabilistic
from n2v.config import config, set_parallel, set_lp_solver, get_config

__all__ = [
    "Star",
    "Zono",
    "Box",
    "ProbabilisticBox",
    "ImageStar",
    "ImageZono",
    "Hexatope",
    "Octatope",
    "HalfSpace",
    "NeuralNetwork",
    "utils",
    "probabilistic",
    "config",
    "set_parallel",
    "set_lp_solver",
    "get_config",
]
