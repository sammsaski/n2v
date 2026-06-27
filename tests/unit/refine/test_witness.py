"""Unit tests for the violation LP, witness feasibility, and infidelity score."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.types import LinearSpec
from n2v.refine.witness import (
    binding_row,
    epsilon_vector,
    is_true_counterexample,
    score_vector,
    violation_lp,
)


def test_binding_row_is_argmax_not_argmin():
    """The active spec row at the epigraph optimum is argmax of the margins."""
    # y = alpha in [-1,1]^2; two spec rows with clearly different margins at a point.
    V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    S = Star(V, np.zeros((0, 2)), np.zeros((0, 1)),
             np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    # row0: y0 - 5  (margin small/negative); row1: y0 + y1 - 0 (larger)
    spec = LinearSpec(G=np.array([[1.0, 0.0], [1.0, 1.0]]), g=np.array([5.0, 0.0]))
    alpha = np.array([0.5, 0.5])
    # margins: row0 = 0.5-5 = -4.5 ; row1 = 1.0-0 = 1.0  -> argmax = row1
    assert binding_row(S, spec, alpha) == 1


def test_faithful_witness_is_feasible_in_P():
    """The faithful witness must satisfy C/d and the box (lies in P)."""
    # y = alpha, alpha in [-1,1]^2, with split constraint alpha0 + alpha1 >= 0.5.
    V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C = np.array([[-1.0, -1.0]])
    d = np.array([[-0.5]])
    S = Star(V, C, d, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    spec = LinearSpec(G=np.array([[1.0, 1.0]]), g=np.array([-0.3]))

    alpha, t = violation_lp(S, spec, include_Cd=True)
    y = S.V[:, 0] + S.V[:, 1:] @ alpha
    assert S.contains(y, method="lp"), "faithful witness fell outside P"
    # min (a0+a1) over P is 0.5, so worst margin t = 0.5 + 0.3 = 0.8 > 0 -> safe.
    assert t == pytest.approx(0.8, abs=1e-6)


def test_box_corner_witness_can_leave_P_and_misjudges_safety():
    """
    The box-corner witness ignores C/d and can fall outside P; here it also
    flips the verdict (box says 'unsafe', P says 'safe'). This is the DRG-BaB
    flaw the constraint-faithful witness fixes (Prop. 2: Delta > 0).
    """
    V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C = np.array([[-1.0, -1.0]])  # alpha0 + alpha1 >= 0.5
    d = np.array([[-0.5]])
    S = Star(V, C, d, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    spec = LinearSpec(G=np.array([[1.0, 1.0]]), g=np.array([-0.3]))

    alpha_f, t_f = violation_lp(S, spec, include_Cd=True)
    alpha_b, t_b = violation_lp(S, spec, include_Cd=False)

    # Box corner minimizes the sum freely -> (-1, -1), which violates C.
    y_b = S.V[:, 0] + S.V[:, 1:] @ alpha_b
    assert not S.contains(y_b, method="lp"), "box corner unexpectedly inside P"
    assert np.any(S.C @ alpha_b > S.d.flatten() + 1e-9)

    # Divergence Delta = t_faithful - t_box >= 0, and strictly positive here.
    delta = t_f - t_b
    assert delta > 1e-6
    # And the verdicts genuinely differ: box thinks unsafe (t_b <= 0), P safe (t_f > 0).
    assert t_b <= 0 < t_f


def test_violation_lp_returns_none_when_predicate_empty():
    """An infeasible predicate polytope (empty set) returns None (vacuously safe)."""
    V = np.array([[0.0, 1.0]])
    # alpha <= -1 AND alpha >= 1 is infeasible.
    C = np.array([[1.0], [-1.0]])
    d = np.array([[-1.0], [-1.0]])
    S = Star(V, C, d, np.array([-5.0]), np.array([5.0]))
    spec = LinearSpec(G=np.array([[1.0]]), g=np.array([0.0]))
    assert violation_lp(S, spec, include_Cd=True) is None


def test_epsilon_nonnegative_and_correct():
    """eps_j >= 0 always, and matches a hand computation on a 1-neuron net."""
    net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[2].bias.zero_()
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0]), np.array([1.0]))
    S_out, meta = relaxed_reach(S_in, layers)

    # Pick alpha = [input=0.5, relaxed_var=0.9]. x_hat = 0.5, max(0,0.5)=0.5,
    # so eps = 0.9 - 0.5 = 0.4.
    alpha = np.array([0.5, 0.9])
    eps = epsilon_vector(alpha, meta)
    assert eps.shape == (1,)
    assert eps[0] == pytest.approx(0.4, abs=1e-9)

    # On many random feasible alphas, eps stays non-negative.
    rng = np.random.default_rng(0)
    for _ in range(200):
        a_in = rng.uniform(-1, 1)
        x_hat = a_in
        # relaxed var must satisfy alpha_r >= max(0, x_hat) by the triangle; sample above it.
        a_r = max(0.0, x_hat) + rng.uniform(0, 0.5)
        assert epsilon_vector(np.array([a_in, a_r]), meta)[0] >= -1e-12


def test_score_uses_influence():
    """score = eps * |h . V_out[:, p(j)]|; zero influence -> zero score."""
    net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[2.0]], dtype=torch.float64))
        net[2].bias.zero_()
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0]), np.array([1.0]))
    S_out, meta = relaxed_reach(S_in, layers)
    alpha = np.array([0.5, 0.9])  # eps = 0.4
    h = np.array([1.0])           # spec direction
    sc, eps = score_vector(alpha, meta, S_out, h)
    # V_out[:, pred_col] is the output coefficient of the relaxed var = 2.0.
    assert sc[0] == pytest.approx(0.4 * 2.0, abs=1e-9)


def test_true_counterexample_detection():
    """A concrete unsafe output is detected as a real CE; a safe one is not."""
    # Identity-ish net: y = relu(x); over x in [-1,1].
    net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[2].bias.zero_()
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0]), np.array([1.0]))
    S_out, meta = relaxed_reach(S_in, layers)
    # unsafe = {y <= -0.5}.  y = relu(x) >= 0, so it's actually UNSAT, but a
    # witness claiming x_in mapping to y < -0.5 must be rejected by the exact pass.
    spec = LinearSpec(G=np.array([[1.0]]), g=np.array([-0.5]))
    # alpha input = 0.8 -> x_in = 0.8 -> y = 0.8, not <= -0.5 -> not a CE.
    is_ce, x_in, y = is_true_counterexample(np.array([0.8, 0.8]), S_in, net, spec)
    assert not is_ce
    assert y[0] == pytest.approx(0.8, abs=1e-9)

    # A genuinely unsafe spec {y <= 1.0} with x_in=0.3 -> y=0.3 <= 1.0 -> real CE.
    spec2 = LinearSpec(G=np.array([[1.0]]), g=np.array([1.0]))
    is_ce2, _, _ = is_true_counterexample(np.array([0.3, 0.3]), S_in, net, spec2)
    assert is_ce2


def test_ce_requires_point_inside_unsafe_region():
    """A point just OUTSIDE the unsafe region is not a counterexample (tol=0)."""
    net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[2].bias.zero_()
    S_in = Star.from_bounds(np.array([-1.0]), np.array([1.0]))
    # x_in = 0.5 -> y = 0.5. unsafe = { y <= 0.5 - 1e-9 } excludes 0.5 by a hair.
    spec = LinearSpec(G=np.array([[1.0]]), g=np.array([0.5 - 1e-9]))
    is_ce, _, y = is_true_counterexample(np.array([0.5, 0.5]), S_in, net, spec)
    assert y[0] == pytest.approx(0.5, abs=1e-12)
    assert not is_ce, "a point outside the unsafe region was accepted as a CE"
    # Move the boundary to include 0.5 exactly -> it IS a CE (closed region).
    spec_in = LinearSpec(G=np.array([[1.0]]), g=np.array([0.5]))
    is_ce2, _, _ = is_true_counterexample(np.array([0.5, 0.5]), S_in, net, spec_in)
    assert is_ce2
