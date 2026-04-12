"""
Flow-based conformal reachability module.

Provides flow-matching-based nonconformity scores for tighter
probabilistic reachable sets than hyperrectangular scores.
"""

from n2v.probabilistic.flow.scores import (
    NonconformityScore,
    HyperrectScore,
    EllipsoidScore,
    BallScore,
    FlowScore,
)
from n2v.probabilistic.flow.calibrate import calibrate, compute_guarantee
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.train import train_flow
from n2v.probabilistic.flow.scenario_verify import (
    sample_truncated_gaussian_ball,
    sample_empirical_latent_ball,
    scenario_verify_halfspace,
    verify_robustness,
    preimage_search,
    ScenarioResult,
    RobustnessResult,
    PreimageResult,
)

__all__ = [
    'NonconformityScore',
    'HyperrectScore',
    'EllipsoidScore',
    'BallScore',
    'FlowScore',
    'calibrate',
    'compute_guarantee',
    'ProbabilisticSet',
    'VelocityField',
    'FlowODE',
    'train_flow',
    'sample_truncated_gaussian_ball',
    'sample_empirical_latent_ball',
    'scenario_verify_halfspace',
    'verify_robustness',
    'preimage_search',
    'ScenarioResult',
    'RobustnessResult',
    'PreimageResult',
]
