"""Soundness for element-wise Mul/Div of two computed sets and Concat
of branches with mismatched predicate systems.

These are the lsnc/mscn/ml4acopf failure classes found by the L3
real-model oracle: McCormick mul and Concat used to zero-pad the
smaller predicate system to the larger one ("prefix alignment"), which
silently identifies UNRELATED predicate variables and can produce an
under-approximation. Both now share predicates only when the systems
are identical and otherwise compose block-diagonally
(_join_star_systems / _join_predicates).

House pattern: build branches with reach_layer, combine, then sample
the input box and assert every true torch output lies inside the reach
set's ranges.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Star
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.reach import _mul_sets, _div_sets, _concat_sets
from n2v.nn.layer_ops import sigmoid_reach


def _assert_contained(model_fns, combine, result, lb_in, ub_in, n=300,
                      seed=7):
    lo, hi = result.estimate_ranges()
    lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = rng.uniform(lb_in, ub_in)
        t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = combine(*[f(t) for f in model_fns]).numpy().flatten()
        assert np.all(y >= lo - 1e-5) and np.all(y <= hi + 1e-5), (
            f"containment violated: y={y}, lo={lo}, hi={hi}")


class TestMulMismatchedPredicates:
    """Element-wise products across branches with different predicate
    systems (the lsnc failure class)."""

    def test_relu_on_one_branch(self):
        """relu(W1 x) * (W2 x): approx-star ReLU appends predicates on
        one branch only. The old zero-padding produced 40/150
        containment violations on lsnc."""
        torch.manual_seed(3)
        W1, W2 = nn.Linear(4, 4), nn.Linear(4, 4)
        relu = nn.ReLU()
        star = Star.from_bounds(np.full(4, -1.0), np.ones(4))

        branch_a = reach_layer(relu, reach_layer(W1, [star], 'approx'),
                               'approx')
        branch_b = reach_layer(W2, [star], 'approx')
        result = _mul_sets(branch_a, branch_b)[0]

        _assert_contained(
            [lambda t: relu(W1(t)), W2], lambda a, b: a * b, result,
            np.full(4, -1.0), np.ones(4))

    def test_relu_on_both_branches(self):
        """relu(W1 x) * relu(W2 x): equal predicate COUNTS are possible
        with different constraints — the count-based trap."""
        torch.manual_seed(4)
        W1, W2 = nn.Linear(4, 4), nn.Linear(4, 4)
        relu = nn.ReLU()
        star = Star.from_bounds(np.full(4, -1.0), np.ones(4))

        branch_a = reach_layer(relu, reach_layer(W1, [star], 'approx'),
                               'approx')
        branch_b = reach_layer(relu, reach_layer(W2, [star], 'approx'),
                               'approx')
        result = _mul_sets(branch_a, branch_b)[0]

        _assert_contained(
            [lambda t: relu(W1(t)), lambda t: relu(W2(t))],
            lambda a, b: a * b, result,
            np.full(4, -1.0), np.ones(4))

    def test_identical_systems_square(self):
        """(W x) * (W x) — identical predicate systems keep the exact
        shared coupling."""
        torch.manual_seed(5)
        W = nn.Linear(4, 4)
        star = Star.from_bounds(np.full(4, -1.0), np.ones(4))
        branch = reach_layer(W, [star], 'approx')
        result = _mul_sets(branch, branch)[0]
        _assert_contained([W], lambda a: a * a, result,
                          np.full(4, -1.0), np.ones(4))


class TestDivSets:
    """Element-wise division of two computed sets (mscn failure class)."""

    def test_div_positive_denominator(self):
        torch.manual_seed(6)
        W1 = nn.Linear(4, 4)
        W2 = nn.Linear(4, 4)
        with torch.no_grad():
            W2.weight.abs_()
            W2.bias.fill_(5.0)  # keeps denominator well above zero
        star = Star.from_bounds(np.full(4, 0.5), np.ones(4))

        num = reach_layer(W1, [star], 'approx')
        den = reach_layer(W2, [star], 'approx')
        result = _div_sets(num, den)[0]

        _assert_contained([W1, W2], lambda a, b: a / b, result,
                          np.full(4, 0.5), np.ones(4))

    def test_div_mismatched_predicates(self):
        """relu numerator (appends predicates) over an affine
        denominator."""
        torch.manual_seed(8)
        W1, W2 = nn.Linear(4, 4), nn.Linear(4, 4)
        relu = nn.ReLU()
        with torch.no_grad():
            W2.weight.abs_()
            W2.bias.fill_(5.0)
        star = Star.from_bounds(np.full(4, -1.0), np.ones(4))

        num = reach_layer(relu, reach_layer(W1, [star], 'approx'),
                          'approx')
        den = reach_layer(W2, [star], 'approx')
        result = _div_sets(num, den)[0]

        _assert_contained([lambda t: relu(W1(t)), W2],
                          lambda a, b: a / b, result,
                          np.full(4, -1.0), np.ones(4))

    def test_zero_straddling_denominator_raises(self):
        star = Star.from_bounds(np.full(2, -1.0), np.ones(2))
        with pytest.raises(NotImplementedError, match="straddles zero"):
            _div_sets([star], [star])


class TestConcatMismatchedPredicates:
    """Concat across branches with different predicate systems: the old
    code kept only one branch's constraints (and prefix-aligned the
    rest), under-approximating the other branches."""

    def test_relu_on_one_branch(self):
        torch.manual_seed(9)
        W1, W2 = nn.Linear(4, 3), nn.Linear(4, 3)
        relu = nn.ReLU()
        star = Star.from_bounds(np.full(4, -1.0), np.ones(4))

        branch_a = reach_layer(relu, reach_layer(W1, [star], 'approx'),
                               'approx')
        branch_b = reach_layer(W2, [star], 'approx')
        result = _concat_sets([branch_a, branch_b], 0)[0]

        _assert_contained(
            [lambda t: relu(W1(t)), W2],
            lambda a, b: torch.cat([a, b], dim=1), result,
            np.full(4, -1.0), np.ones(4))

    def test_identical_systems_stay_exact(self):
        """concat(W1 x, W2 x): same predicate system on both branches —
        the result must stay EXACT (width of the concat equals the
        widths of the branches)."""
        torch.manual_seed(10)
        W1, W2 = nn.Linear(3, 2), nn.Linear(3, 2)
        star = Star.from_bounds(np.full(3, -1.0), np.ones(3))
        a = reach_layer(W1, [star], 'approx')
        b = reach_layer(W2, [star], 'approx')
        result = _concat_sets([a, b], 0)[0]

        lo, hi = result.get_ranges()
        la, ha = a[0].get_ranges()
        lb_, hb = b[0].get_ranges()
        np.testing.assert_allclose(
            np.asarray(lo).flatten(),
            np.concatenate([np.asarray(la).flatten(),
                            np.asarray(lb_).flatten()]), atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(hi).flatten(),
            np.concatenate([np.asarray(ha).flatten(),
                            np.asarray(hb).flatten()]), atol=1e-8)


class TestSigmoidConstantStar:
    """Regression: a star can carry its (constant) value in constrained
    predicate variables with a ZERO center column — e.g. the z vars of
    a McCormick product. The all-constant branch used to apply the
    function to the center column and returned sigmoid(0) = 0.5
    regardless of the true value (mscn, dev 2.3e-01)."""

    def test_value_in_predicate_not_center(self):
        val = 0.98
        V = np.array([[0.0, 1.0]])  # x = 0 + 1 * alpha
        s = Star(V, np.zeros((0, 1)), np.zeros((0, 1)),
                 np.array([[val]]), np.array([[val]]))  # alpha == val
        out = sigmoid_reach.sigmoid_star_approx([s], lp_solver='linprog')[0]
        lo, hi = out.get_ranges()
        expected = 1.0 / (1.0 + np.exp(-val))
        np.testing.assert_allclose(np.asarray(lo).flatten(), [expected],
                                   atol=1e-9)
        np.testing.assert_allclose(np.asarray(hi).flatten(), [expected],
                                   atol=1e-9)
