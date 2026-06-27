"""
Soundness of the Phase-1 refinement BaB.

The critical claims under test:

  * No false UNSAT, and completeness on small nets: the BaB verdict equals a
    trusted oracle (n2v's exact star reach) on random instances, under EVERY
    selector. Selection cannot change the verdict -- only the node count.
  * Honest counterexamples (Theorem 1): every SAT carries a concrete input the
    exact model maps into the unsafe region.
  * Localization (Theorem 1, contrapositive): a fully-fixed (exact) leaf whose
    witness has eps == 0 everywhere is a real counterexample.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.star import Star
from n2v.utils.lpsolver import check_feasibility
from n2v.refine import (
    BoundWidthSelector,
    BoxCornerSelector,
    FaithfulSelector,
    LinearSpec,
    RandomSelector,
    Status,
    verify_refine,
)


# --------------------------------------------------------------------------- #
# Trusted oracle: exact star reach + per-star unsafe-intersection feasibility. #
# --------------------------------------------------------------------------- #
def _star_intersects_unsafe(S: Star, spec: LinearSpec) -> bool:
    c_out = S.V[:, 0]
    V_out = S.V[:, 1:]
    A_rows = [spec.G @ V_out]
    b_rows = [spec.g - spec.G @ c_out]
    if S.C.size > 0:
        A_rows.append(S.C)
        b_rows.append(S.d.flatten())
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    return check_feasibility(
        A=A, b=b, lb=S.predicate_lb.flatten(), ub=S.predicate_ub.flatten()
    )


def _oracle_verdict(model, input_star: Star, spec: LinearSpec) -> Status:
    """Sound+complete verdict via exact reach."""
    out_stars = reach_pytorch_model(model, input_star, method="exact")
    for S in out_stars:
        if _star_intersects_unsafe(S, spec):
            return Status.SAT
    return Status.UNSAT


def _rand_net(seed, dims=(2, 5, 5, 2)):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers).double()


ALL_SELECTORS = [
    FaithfulSelector,
    BoxCornerSelector,
    BoundWidthSelector,
    lambda: RandomSelector(seed=47),
]


@pytest.mark.parametrize("seed", range(8))
def test_verdict_matches_exact_oracle_all_selectors(seed):
    """BaB verdict == exact-reach oracle, for every selector (sound + complete)."""
    rng = np.random.default_rng(seed)
    net = _rand_net(seed)
    lb = rng.uniform(-1.0, -0.2, size=2)
    ub = rng.uniform(0.2, 1.0, size=2)
    S_in = Star.from_bounds(lb, ub)

    # Pick a spec that lands near the output range so we get a mix of SAT/UNSAT:
    # unsafe = { y_0 <= tau }. Use the exact-reach midpoint as a tunable threshold.
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mins = min(float(S.get_range(0)[0]) for S in out0)
    maxs = max(float(S.get_range(0)[1]) for S in out0)
    tau = mins + (maxs - mins) * (0.25 + 0.5 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle_verdict(net, S_in, spec)

    for make in ALL_SELECTORS:
        res = verify_refine(S_in, net, spec, make(), node_budget=20000)
        assert res.status != Status.UNKNOWN, "ran out of budget on a tiny net"
        assert res.status == oracle, (
            f"selector {make().name}: BaB={res.status} oracle={oracle} (seed={seed})"
        )
        if res.status == Status.SAT:
            # Theorem 1 honesty: returned CE truly violates the property.
            assert res.counterexample_x is not None
            with torch.no_grad():
                y = net(torch.as_tensor(res.counterexample_x, dtype=torch.float64))
            assert spec.is_unsafe(y.numpy().flatten()), "returned a fake CE"


@pytest.mark.parametrize("seed", range(8))
def test_no_false_unsat_against_sampling(seed):
    """If dense sampling finds a counterexample, the BaB must NOT say UNSAT."""
    rng = np.random.default_rng(1000 + seed)
    net = _rand_net(seed + 100)
    lb = rng.uniform(-1.0, -0.2, size=2)
    ub = rng.uniform(0.2, 1.0, size=2)
    S_in = Star.from_bounds(lb, ub)

    X = rng.uniform(lb, ub, size=(40000, 2))
    with torch.no_grad():
        Y = net(torch.as_tensor(X, dtype=torch.float64)).numpy()
    # unsafe = { y_0 <= tau } with tau chosen so a CE provably exists.
    tau = float(np.quantile(Y[:, 0], 0.10))
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
    assert np.any(Y[:, 0] <= tau), "test setup: sampling found no CE"

    for make in ALL_SELECTORS:
        res = verify_refine(S_in, net, spec, make(), node_budget=20000)
        assert res.status == Status.SAT, (
            f"false non-SAT ({res.status}) by {make().name} despite sampled CE (seed={seed})"
        )


def test_returned_counterexample_lies_in_input_box():
    """Sanity: the CE input respects the input set bounds."""
    rng = np.random.default_rng(5)
    net = _rand_net(7)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    # A loose spec that is almost surely SAT: y_0 <= large.
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([1e3]))
    res = verify_refine(S_in, net, spec, FaithfulSelector())
    assert res.status == Status.SAT
    assert np.all(res.counterexample_x >= -1.0 - 1e-9)
    assert np.all(res.counterexample_x <= 1.0 + 1e-9)


def test_theorem1_zero_epsilon_leaf_is_real_ce():
    """
    A fully-fixed (exact) leaf that is feasible-in-unsafe must be a real CE
    (Theorem 1): with no relaxed neurons, eps == 0 everywhere by definition, and
    the witness reproduces the exact output.
    """
    from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
    from n2v.refine.types import NeuronKey, Phase
    from n2v.refine.witness import epsilon_vector, is_true_counterexample, violation_lp

    net = _rand_net(3, dims=(2, 4, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))

    # Fix every hidden neuron ACTIVE -> exact (no relaxation).
    fixed = {NeuronKey(0, j): Phase.ACTIVE for j in range(4)}
    S_out, meta = relaxed_reach(S_in, layers, fixed)
    assert meta == []  # exact star: nothing relaxed

    # A spec guaranteed to intersect this exact star: y_0 <= its own max.
    ymax = float(S_out.get_range(0)[1])
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([ymax]))
    res = violation_lp(S_out, spec, include_Cd=True)
    assert res is not None
    alpha, t = res
    assert t <= 1e-7  # feasible in unsafe
    assert epsilon_vector(alpha, meta).size == 0  # eps vacuously zero
    is_ce, x_in, y = is_true_counterexample(alpha, S_in, net, spec)
    assert is_ce, "Theorem 1 violated: exact feasible leaf is not a real CE"
