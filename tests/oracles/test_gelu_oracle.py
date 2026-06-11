"""Pushforward-containment oracle for GELU reachability."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import gelu_reach

from tests.oracles import assert_set_contains_pushforward


def _concrete_gelu(x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return nn.GELU()(torch.from_numpy(x).double()).detach().cpu().numpy()


def test_gelu_box_oracle_covers_negative_dip():
    """GELU's global minimum is at x ≈ -0.752; the box reach must include it."""
    lb = np.array([[-2.0]])
    ub = np.array([[2.0]])
    inp = Box(lb, ub)
    out = gelu_reach.gelu_box([inp])
    assert_set_contains_pushforward(_concrete_gelu, inp, out, n_samples=512)
    # Hand check: output lower bound must be at or below the global minimum.
    assert out[0].lb.item() <= -0.16


def test_gelu_star_oracle_preserves_image_shape():
    """A small box-shaped Star pushed through GELU should still be contained."""
    lb = np.array([[-1.5], [0.0], [1.0], [-0.5]])
    ub = np.array([[-0.5], [1.0], [2.0], [0.5]])
    inp = Star.from_bounds(lb, ub)
    out = gelu_reach.gelu_star_approx([inp])
    assert_set_contains_pushforward(_concrete_gelu, inp, out, n_samples=256)


def test_gelu_zono_oracle():
    lb = np.array([[-1.5], [0.0], [1.0], [-0.5]])
    ub = np.array([[-0.5], [1.0], [2.0], [0.5]])
    inp = Zono.from_bounds(lb, ub)
    out = gelu_reach.gelu_zono([inp])
    assert_set_contains_pushforward(_concrete_gelu, inp, out, n_samples=256)
