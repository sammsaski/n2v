"""Shared reach-set helpers for the FlowConformal benchmarks.

Currently only hosts :func:`compute_exact_reach`, a thin wrapper around
:meth:`n2v.nn.NeuralNetwork.reach` that falls back from ``method='approx'``
to ``method='exact'`` when the approximate propagation returns a single
loose box. Used by every benchmark harness that needs a Star-union
ground truth.

Previously lived inside
``examples/FlowConformal/experiments/hashemi_comparison/v2_adaptive_reg/exp_hashemi_comparison.py``
under the ``v2.`` namespace.
"""
from __future__ import annotations

import time

import numpy as np

from n2v.nn import NeuralNetwork
from n2v.sets.box import Box


def compute_exact_reach(
    net,
    x_center_np: np.ndarray,
    radius: float,
    output_dim: int,
) -> dict:
    """Propagate the L-infinity input box through ``net`` with n2v's Star
    reach, preferring the approximate path and falling back to exact.

    Args:
        net: Module with a ``.net`` attribute exposing a ``nn.Sequential``
            (the n2v convention used throughout FlowConformal).
        x_center_np: ``(input_dim,)`` numpy array giving the input-ball center.
        radius: Half-width of the L-infinity input ball.
        output_dim: Unused; kept for backward-compat with existing callers.

    Returns:
        Dict with keys:
            ``stars``: list of :class:`n2v.sets.star.Star` (possibly a
                single approx zonotope-like Star, possibly thousands of exact
                polytopes).
            ``method``: ``'approx'``, ``'exact'``, or ``'approx-failed'`` /
                ``'error:<msg>'`` on failure.
            ``time``: wall-clock seconds.
    """
    lb = (x_center_np - radius).reshape(-1, 1)
    ub = (x_center_np + radius).reshape(-1, 1)
    input_star = Box(lb, ub).to_star()
    wrapper = NeuralNetwork(net.net)
    t0 = time.time()
    try:
        stars = wrapper.reach(input_star, method='approx')
        method = 'approx'
    except Exception:
        stars = None
        method = 'approx-failed'
    # Fall back to exact if approx returns a single loose box.
    if stars is None or len(stars) == 1:
        try:
            stars = wrapper.reach(input_star, method='exact')
            method = 'exact'
        except Exception as e:
            return {'stars': [], 'method': f'error:{e}', 'time': time.time() - t0}
    return {
        'stars': stars,
        'method': method,
        'time': time.time() - t0,
    }
