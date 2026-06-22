"""
n2v: Neural Network Verification Tool for Python/PyTorch

A formal verification tool for deep learning models, supporting reachability
analysis and robustness verification using set-based methods.

Translated from the original MATLAB NNV tool by the VeriVital research group.
"""

__version__ = "0.1.0"
__author__ = "NNV Team"

from n2v.sets import Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope, HalfSpace, ProbabilisticBox
from n2v.nn import NeuralNetwork, ReachConfig
from n2v.probabilistic import (
    ConformalReachConfig,
    FlowReachConfig,
    ProbabilisticSet,
    conformal_reach,
    flow_reach,
)
from n2v import utils
from n2v import probabilistic
from n2v.config import config, set_parallel, set_lp_solver, get_config
from n2v.utils.lp_solver_enum import Backend, LPSolver

__all__ = [
    # Sets
    "Star",
    "Zono",
    "Box",
    "ProbabilisticBox",
    "ProbabilisticSet",
    "ImageStar",
    "ImageZono",
    "Hexatope",
    "Octatope",
    "HalfSpace",
    # OO reach
    "NeuralNetwork",
    "ReachConfig",
    # Probabilistic reach (model-agnostic free functions + configs)
    "flow_reach",
    "FlowReachConfig",
    "conformal_reach",
    "ConformalReachConfig",
    # Sub-packages
    "utils",
    "probabilistic",
    # Config
    "config",
    "set_parallel",
    "set_lp_solver",
    "get_config",
    "Backend",
    "LPSolver",
]
