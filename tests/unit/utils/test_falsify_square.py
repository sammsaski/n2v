"""
Unit tests for the self-contained gradient-free Square attack (method='square')
and the 'strong' ensemble (random -> APGD -> Square).

The Square attack is a random-search falsifier that needs no gradients, so it
attacks models where PGD/APGD fail because gradients vanish (e.g. Sign/binarized
activations). Every SAT it returns is verified against the canonical
``_output_satisfies_property`` check, so a SAT result is always sound.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.utils import falsify
from n2v.utils.falsify import (
    METHODS,
    _build_group_arrays,
    _batch_total_margin,
    _extract_halfspace_groups,
    _output_satisfies_property,
)
from n2v.sets import HalfSpace


class _SignNet(nn.Module):
    """y = sign(x @ W.T). Non-differentiable: gradient is 0 a.e., so PGD/APGD
    cannot climb toward the unsafe region — the motivating case for Square."""

    def __init__(self, W):
        super().__init__()
        self.lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
        self.lin.weight.data = torch.tensor(W, dtype=torch.float32)

    def forward(self, x):
        return torch.sign(self.lin(x))


class TestBatchTotalMargin:
    """The batched margin proxy must agree in sign with the canonical
    ``_output_satisfies_property`` check — the soundness-critical invariant
    (margin <= 0  iff  the point is in the unsafe region)."""

    def _check_agreement(self, prop, outs):
        groups = _extract_halfspace_groups(prop)
        garr = _build_group_arrays(groups)
        margins = _batch_total_margin(np.asarray(outs, dtype=np.float32), garr)
        for y, m in zip(outs, margins):
            canonical = _output_satisfies_property(np.asarray(y, dtype=np.float32), groups)
            assert (m <= 0) == canonical, (
                f"margin/canonical disagree at y={y}: margin={m}, canonical={canonical}"
            )

    def test_single_halfspace(self):
        # Unsafe: y0 >= 0.5  (-y0 <= -0.5)
        prop = [{'Hg': HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))}]
        self._check_agreement(prop, [[1.0, 0.0], [0.5, 0.0], [0.49, 9.0], [-3.0, 0.0]])

    def test_multi_row_halfspace_and(self):
        # One halfspace with two rows (AND of rows): y0 >= 0.5 AND y1 >= 0.5
        G = np.array([[-1.0, 0.0], [0.0, -1.0]])
        g = np.array([[-0.5], [-0.5]])
        prop = [{'Hg': HalfSpace(G, g)}]
        self._check_agreement(prop, [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.6]])

    def test_multi_group_and(self):
        # Two groups ANDed: (y0 >= 0.5) AND (y1 >= 0.5)
        prop = [
            {'Hg': HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))},
            {'Hg': HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.5]]))},
        ]
        self._check_agreement(prop, [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.49, 0.51]])

    def test_or_within_group(self):
        # One group, OR of two halfspaces: y0 >= 0.8  OR  y0 <= 0.2
        hs_a = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.8]]))
        hs_b = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.2]]))
        prop = [{'Hg': [hs_a, hs_b]}]
        self._check_agreement(prop, [[0.9, 0.0], [0.1, 0.0], [0.5, 0.0], [0.8, 0.0]])


class TestFalsifySquare:
    """Tests for the Square attack (method='square')."""

    def test_finds_counterexample(self):
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        # Unsafe: output[0] >= 0.5
        hs = HalfSpace(np.array([[-1.0, 0.0]]), np.array([-0.5]))

        result, cex = falsify(model, lb, ub, hs, method='square',
                              n_iters=2000, batch=64, seed=42)

        assert result == 0, "Should find counterexample (SAT)"
        assert cex is not None
        inp, out = cex
        # Sound: the returned point really is in the unsafe region.
        groups = _extract_halfspace_groups(hs)
        assert _output_satisfies_property(np.asarray(out, dtype=np.float32), groups)
        assert inp[0] >= 0.5 - 1e-6

    def test_no_counterexample(self):
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)
        lb = np.array([0.0, 0.0])
        ub = np.array([0.4, 0.4])  # inputs never reach 0.5
        hs = HalfSpace(np.array([[-1.0, 0.0]]), np.array([-0.5]))

        result, cex = falsify(model, lb, ub, hs, method='square',
                              n_iters=2000, batch=64, seed=42)

        assert result == 2, "Should return UNKNOWN (no counterexample exists)"
        assert cex is None

    def test_with_relu_model(self):
        model = nn.Sequential(
            nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False)
        )
        model[0].weight.data = torch.eye(2)
        model[2].weight.data = torch.ones(1, 2)
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        # Unsafe: output >= 1.5
        hs = HalfSpace(np.array([[-1.0]]), np.array([-1.5]))

        result, cex = falsify(model, lb, ub, hs, method='square',
                              n_iters=3000, batch=64, seed=42)

        assert result == 0, "Should find counterexample"
        _, out = cex
        assert out[0] >= 1.5 - 1e-6

    def test_finds_on_sign_model_zero_gradient(self):
        """The motivating case: a non-differentiable Sign model. Gradients
        vanish (PGD/APGD cannot climb), but gradient-free Square finds the CE."""
        model = _SignNet(np.array([[1.0, 1.0]]))  # y = sign(x0 + x1)
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        # Unsafe: output >= 0.5  (i.e. sign(...) == 1, reachable when x0+x1 > 0)
        hs = HalfSpace(np.array([[-1.0]]), np.array([-0.5]))

        result, cex = falsify(model, lb, ub, hs, method='square',
                              n_iters=4000, batch=64, seed=0)

        assert result == 0, "Square should crack the zero-gradient Sign model"
        _, out = cex
        assert out[0] >= 0.5 - 1e-6

    def test_respects_multi_group_and_logic(self):
        """Soundness: a point satisfying only one of two ANDed groups is NOT a CE."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        # Group 0: x0 >= 0.3 (feasible);  Group 1: x1 >= 2.0 (infeasible in box)
        prop = [
            {'Hg': HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.3]]))},
            {'Hg': HalfSpace(np.array([[0.0, -1.0]]), np.array([[-2.0]]))},
        ]

        result, cex = falsify(model, lb, ub, prop, method='square',
                              n_iters=3000, batch=64, seed=42)

        assert result == 2, "No input satisfies both groups -> UNKNOWN, not a false SAT"
        assert cex is None

    def test_reproducible_with_seed(self):
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        hs = HalfSpace(np.array([[-1.0, 0.0]]), np.array([-0.5]))

        r1, c1 = falsify(model, lb, ub, hs, method='square', n_iters=500, batch=32, seed=123)
        r2, c2 = falsify(model, lb, ub, hs, method='square', n_iters=500, batch=32, seed=123)

        assert r1 == r2
        if c1 is not None and c2 is not None:
            np.testing.assert_array_equal(c1[0], c2[0])
            np.testing.assert_array_equal(c1[1], c2[1])

    def test_ignores_extra_kwargs(self):
        """Square must tolerate kwargs from the 'strong' ensemble (n_samples, etc.)."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        hs = HalfSpace(np.array([[-1.0, 0.0]]), np.array([-0.5]))

        result, _ = falsify(model, lb, ub, hs, method='square',
                            n_iters=1000, batch=32, n_samples=10,
                            n_restarts=3, n_steps=5, seed=42)
        assert result == 0


class TestFalsifyStrong:
    """Tests for the self-contained 'strong' ensemble (random -> APGD -> Square)."""

    def test_square_and_strong_registered(self):
        assert 'square' in METHODS
        assert 'strong' in METHODS

    def test_finds_counterexample(self):
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        hs = HalfSpace(np.array([[-1.0, 0.0]]), np.array([-0.5]))

        result, cex = falsify(model, lb, ub, hs, method='strong',
                              n_samples=100, n_restarts=3, n_steps=10,
                              n_iters=1000, batch=64, seed=42)

        assert result == 0
        assert cex is not None

    def test_falls_through_to_square_on_sign_model(self):
        """With random disabled (n_samples=0) and gradients vanishing, the
        ensemble must still succeed — Square is the backstop."""
        model = _SignNet(np.array([[1.0, 1.0]]))
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        hs = HalfSpace(np.array([[-1.0]]), np.array([-0.5]))

        result, cex = falsify(model, lb, ub, hs, method='strong',
                              n_samples=0, n_restarts=2, n_steps=10,
                              n_iters=4000, batch=64, seed=0)

        assert result == 0, "Strong ensemble should crack the Sign model via Square"
        assert cex is not None

    def test_no_counterexample(self):
        """Soundness: ensemble returns UNKNOWN (never a false SAT) when no CE exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)
        lb = np.array([0.0, 0.0])
        ub = np.array([0.4, 0.4])
        hs = HalfSpace(np.array([[-1.0, 0.0]]), np.array([-0.5]))

        result, cex = falsify(model, lb, ub, hs, method='strong',
                              n_samples=200, n_restarts=3, n_steps=20,
                              n_iters=2000, batch=64, seed=42)

        assert result == 2
        assert cex is None
