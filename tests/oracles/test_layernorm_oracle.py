"""Pushforward-containment oracle for LayerNorm reachability."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import layernorm_reach

from tests.oracles import assert_set_contains_pushforward


def _concrete_layernorm(layer: nn.LayerNorm):
    layer.eval()

    def _apply(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y = layer.double()(torch.from_numpy(x).double())
        return y.detach().cpu().numpy()

    return _apply


def test_layernorm_box_oracle():
    layer = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    inp = Box(lb, ub)
    out = layernorm_reach.layernorm_box(layer, [inp])
    assert_set_contains_pushforward(
        _concrete_layernorm(layer), inp, out, n_samples=256
    )


def test_layernorm_star_oracle_preserves_predicates():
    layer = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    inp = Star.from_bounds(lb, ub)
    out = layernorm_reach.layernorm_star_approx(layer, [inp])
    assert_set_contains_pushforward(
        _concrete_layernorm(layer), inp, out, n_samples=256
    )
    # Predicate-preservation contract: output Star should carry the input's
    # predicate basis through plus per-feature slack predicates.
    assert out[0].nVar >= inp.nVar, "LayerNorm star reach must not drop predicates"


def test_layernorm_zono_oracle():
    layer = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    inp = Zono.from_bounds(lb, ub)
    out = layernorm_reach.layernorm_zono(layer, [inp])
    assert_set_contains_pushforward(
        _concrete_layernorm(layer), inp, out, n_samples=256
    )
