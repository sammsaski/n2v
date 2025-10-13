"""
Reachability analysis methods for neural networks.

PREFERRED USAGE:
    Use the unified interface via NeuralNetwork.reach() or the set's reach() method:

    >>> from n2v.nn import NeuralNetwork
    >>> from n2v.sets import Star
    >>> net = NeuralNetwork(model)
    >>> input_star = Star.from_bounds(lb, ub)
    >>> output_stars = net.reach(input_star, method='exact')

    Or use the set's own reach() method:
    >>> output_stars = input_star.reach(model, method='exact')

LEGACY FUNCTIONS:
    The functions below (reach_star_exact, reach_star_approx, etc.) are still
    available for backward compatibility, but the new unified interface is
    recommended for all new code.
"""

from n2v.nn.reach.reach_star import reach_star_exact, reach_star_approx
from n2v.nn.reach.reach_zono import reach_zono_approx
from n2v.nn.reach.reach_box import reach_box_approx
from n2v.nn.reach.reach_hexatope import reach_hexatope_approx, reach_hexatope_exact
from n2v.nn.reach.reach_octatope import reach_octatope_approx, reach_octatope_exact
from n2v.nn.reach.dispatcher import reach_pytorch_model

__all__ = [
    "reach_star_exact",
    "reach_star_approx",
    "reach_zono_approx",
    "reach_box_approx",
    "reach_hexatope_approx",
    "reach_hexatope_exact",
    "reach_octatope_approx",
    "reach_octatope_exact",
    "reach_pytorch_model",
]
