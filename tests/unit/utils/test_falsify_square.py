"""
Unit tests for the self-contained gradient-free Square attack (method='square')
and the 'strong' ensemble (random -> APGD -> Square).

The Square attack is a random-search falsifier that needs no gradients, so it
attacks models where PGD/APGD fail because gradients vanish (e.g. Sign/binarized
activations). Every SAT it returns is verified against the canonical
``_output_satisfies_property`` check, so a SAT result is always sound.
"""

from fractions import Fraction

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


class _Recorder(nn.Module):
    """Runs ``inner`` and keeps every input row (as float64) and the batch
    size of every forward."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.rows = []
        self.batches = []

    def forward(self, x):
        self.rows.extend(x.detach().reshape(x.shape[0], -1).to(torch.float64).numpy().copy())
        self.batches.append(x.shape[0])
        return self.inner(x)


class TestBatchTotalMargin:
    """The batched margin proxy must agree in sign with the canonical
    ``_output_satisfies_property`` check — the soundness-critical invariant
    (margin <= 0  iff  the point is in the unsafe region). Both are checked
    against the reading of the property written by hand beside each point
    (repaired 2026-11-18: the two n2v helpers were only compared with each
    other, which passes when both read the property the same wrong way)."""

    def _check_agreement(self, prop, outs, unsafe):
        groups = _extract_halfspace_groups(prop)
        garr = _build_group_arrays(groups)
        margins = _batch_total_margin(np.asarray(outs, dtype=np.float32), garr)
        for y, m, expected in zip(outs, margins, unsafe):
            canonical = _output_satisfies_property(np.asarray(y, dtype=np.float32), groups)
            assert canonical == expected, (
                f"canonical check reads y={y} as unsafe={canonical}, the property says {expected}")
            assert (m <= 0) == expected, (
                f"margin {m} at y={y} reads unsafe={m <= 0}, the property says {expected}")

    def test_single_halfspace(self):
        # Unsafe: y0 >= 0.5  (-y0 <= -0.5); 0.5 itself is unsafe (closed)
        prop = [{'Hg': HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))}]
        self._check_agreement(prop, [[1.0, 0.0], [0.5, 0.0], [0.49, 9.0], [-3.0, 0.0]],
                              [True, True, False, False])

    def test_multi_row_halfspace_and(self):
        # One halfspace with two rows (AND of rows): y0 >= 0.5 AND y1 >= 0.5
        G = np.array([[-1.0, 0.0], [0.0, -1.0]])
        g = np.array([[-0.5], [-0.5]])
        prop = [{'Hg': HalfSpace(G, g)}]
        self._check_agreement(prop, [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.6]],
                              [True, False, False, True])

    def test_multi_group_and(self):
        # Two groups ANDed: (y0 >= 0.5) AND (y1 >= 0.5)
        prop = [
            {'Hg': HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))},
            {'Hg': HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.5]]))},
        ]
        self._check_agreement(prop, [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.49, 0.51]],
                              [True, False, False, False])

    def test_or_within_group(self):
        # One group, OR of two halfspaces: y0 >= 0.8  OR  y0 <= 0.2
        hs_a = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.8]]))
        hs_b = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.2]]))
        prop = [{'Hg': [hs_a, hs_b]}]
        self._check_agreement(prop, [[0.9, 0.0], [0.1, 0.0], [0.5, 0.0], [0.8, 0.0]],
                              [True, True, False, True])


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
        # Sound: the returned point is in the box and, replayed on y = x, in
        # the unsafe region, exactly (repaired 2026-11-18: the check used
        # n2v's own property checker and allowed 1e-6 below the boundary).
        inp = np.asarray(inp, dtype=np.float64)
        assert np.all(lb <= inp) and np.all(inp <= ub)
        assert inp[0] >= 0.5 and out[0] == inp[0]

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
        # Replayed in rationals: relu(x0) + relu(x1) >= 3/2, exactly
        # (repaired 2026-11-18: the check read the returned output with 1e-6
        # of slack below the boundary).
        x = [Fraction(float(v)) for v in np.asarray(cex[0], dtype=np.float64)]
        assert all(lo <= v <= hi for v, lo, hi in zip(x, lb, ub))
        assert max(x[0], 0) + max(x[1], 0) >= Fraction(3, 2)

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
        # Replayed in rationals: sign(x0 + x1) = 1, i.e. x0 + x1 > 0 exactly
        # (repaired 2026-11-18: the check allowed 1e-6 of slack).
        x = [Fraction(float(v)) for v in np.asarray(cex[0], dtype=np.float64)]
        assert all(lo <= v <= hi for v, lo, hi in zip(x, lb, ub))
        assert x[0] + x[1] > 0

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

    # Added 2026-11-18. The models are dyadic, so the unsafe set is known by
    # hand and the witness is replayed in rationals; the seed is fixed (and the
    # autouse fixture seeds the global generators) and the run is on the CPU.

    def test_finds_the_one_unsafe_corner_of_a_mixed_box(self):
        """y0 = W0 . x + 1/2 over a box of intervals of different widths and
        signs with one degenerate coordinate reaches 11.125 at one corner only
        (ub where the weight is positive, lb where it is negative), with margin
        exactly 0 there. Square moves one coordinate per candidate (6 inputs)
        to an extreme or inside the box and keeps every improvement, so the
        corner is reached by moves to the extremes; at the corner the margin is
        0, a hit of the closed region. Catches a hit test that needs a negative
        margin (no later candidate improves on 0) and extremes drawn from one
        side of the box only (the corner needs both)."""
        W = [[1.0, -2.0, 0.5, 4.0, -1.0, 0.25], [0.5, 0.25, -1.0, 0.0, 2.0, -0.5]]
        b = [0.5, -1.0]
        model = nn.Linear(6, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor(W))
            model.bias.copy_(torch.tensor(b))
        lb = np.array([-2.0, -0.5, 1.0, 0.75, -3.0, 0.0])
        ub = np.array([2.0, 0.25, 3.0, 0.75, -1.0, 0.5])
        hs = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-11.125]]))
        result, cex = falsify(model, lb, ub, hs, method='square', seed=3,
                              n_iters=2000, batch=64)
        assert result == 0, "the unsafe corner was not found"
        x = np.asarray(cex[0], dtype=np.float64)
        assert np.all(lb <= x) and np.all(x <= ub)
        y0 = sum(Fraction(w) * Fraction(float(v)) for w, v in zip(W[0], x)) + Fraction(b[0])
        assert y0 >= Fraction(11.125), f"witness {x} is not unsafe"

    def test_spends_exactly_n_iters_in_batches_of_at_most_batch(self):
        """No counterexample exists, so the search runs to the end of its
        budget: the model sees n_iters = 40 points in all (the start point
        counts, the last batch is clamped; 40 - 1 is no multiple of 16) and
        never more than batch = 16 in one forward. Catches a last batch that
        is not clamped (49 points), an ignored ``batch`` and an ignored
        ``n_iters`` (8000 points)."""
        inner = nn.Linear(2, 1)
        with torch.no_grad():
            inner.weight.fill_(1.0)
            inner.bias.zero_()
        model = _Recorder(inner)
        hs = HalfSpace(np.array([[-1.0]]), np.array([[-100.0]]))
        result = falsify(model, np.array([-1.0, 0.5]), np.array([2.0, 0.75]),
                         hs, method='square', seed=3, n_iters=40, batch=16)
        assert result == (2, None)
        assert sum(model.batches) == 40, f"{sum(model.batches)} evaluations"
        assert max(model.batches) <= 16, f"forwards of {model.batches} rows"

    @pytest.mark.parametrize("p_init, low, high", [(0.75, 6.5, 7.5), (0.05, 1, 1)])
    def test_first_blocks_hold_the_p_init_fraction_of_the_coordinates(self, p_init, low, high):
        """p_init is the fraction of the d = 10 coordinates that the first
        candidates perturb (a block of at least one). The start point is an
        interior random point and every perturbed coordinate gets a box extreme
        or a fresh uniform value, so a candidate of the first batch differs
        from the start in exactly its block: about p_init * d = 7.5 (the
        fraction has decayed by one evaluation of n_iters = 1000), and 1 for
        p_init = 0.05 (0.5 coordinates, floored to the block of one). Catches
        an ignored ``p_init`` (the default 0.3 gives 2) and an empty block."""
        inner = nn.Linear(10, 1)
        with torch.no_grad():
            inner.weight.fill_(1.0)
            inner.bias.zero_()
        model = _Recorder(inner)
        hs = HalfSpace(np.array([[-1.0]]), np.array([[-100.0]]))
        lb, ub = np.full(10, -1.25), np.full(10, 2.0)
        result = falsify(model, lb, ub, hs, method='square', seed=3,
                         n_iters=1000, batch=8, p_init=p_init)
        assert result == (2, None)
        start, first_batch = model.rows[0], model.rows[1:9]
        changed = [int(np.sum(c != start)) for c in first_batch]
        assert all(low <= n <= high for n in changed), (
            f"p_init={p_init}: the first candidates change {changed} of 10 coordinates")


class TestFalsifyStrong:
    """Tests for the self-contained 'strong' ensemble (random -> APGD -> Square)."""

    def test_square_and_strong_registered(self):
        assert 'square' in METHODS
        assert 'strong' in METHODS

    def test_random_square_registered(self):
        assert 'random+square' in METHODS

    def test_random_square_falls_through_to_square_on_sign(self):
        """'random+square' (random -> Square): with the random leg disabled
        (n_samples=0), the cascade reaches Square and cracks the zero-gradient
        Sign model where PGD/APGD would be wasted."""
        model = _SignNet(np.array([[1.0, 1.0]]))
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        hs = HalfSpace(np.array([[-1.0]]), np.array([-0.5]))
        result, cex = falsify(model, lb, ub, hs, method='random+square',
                              n_samples=0, n_iters=4000, batch=64, seed=0)
        assert result == 0, "random+square should crack the Sign model via Square"
        assert cex is not None

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
