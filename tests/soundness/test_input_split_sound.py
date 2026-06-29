"""
Soundness of input-space branch-and-bound (n2v.refine.input_split).

Same contract as the activation-split engine: the verdict must equal a trusted
oracle (exact star reach), the branching axis (input bisection) only affects the
node count, and every SAT carries a concrete input the exact model maps into the
unsafe region. No false UNSAT.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.star import Star
from n2v.utils.lpsolver import check_feasibility
from n2v.refine.input_split import verify_refine_input
from n2v.refine.types import LinearSpec, Status


def _star_intersects_unsafe(S: Star, spec: LinearSpec) -> bool:
    c_out = S.V[:, 0]; V_out = S.V[:, 1:]
    A_rows = [spec.G @ V_out]; b_rows = [spec.g - spec.G @ c_out]
    if S.C.size > 0:
        A_rows.append(S.C); b_rows.append(S.d.flatten())
    return check_feasibility(A=np.vstack(A_rows), b=np.concatenate(b_rows),
                             lb=S.predicate_lb.flatten(), ub=S.predicate_ub.flatten())


def _oracle(model, S_in, spec):
    for S in reach_pytorch_model(model, S_in, method="exact"):
        if _star_intersects_unsafe(S, spec):
            return Status.SAT
    return Status.UNSAT


def _rand_net(seed, dims=(3, 6, 6, 2)):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers).double()


@pytest.mark.parametrize("heuristic", ["smear", "widest"])
@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu"])
@pytest.mark.parametrize("seed", range(6))
def test_input_split_matches_oracle(seed, bound_mode, heuristic):
    rng = np.random.default_rng(seed)
    net = _rand_net(seed)
    lb = rng.uniform(-1.0, -0.2, size=3); ub = rng.uniform(0.2, 1.0, size=3)
    S_in = Star.from_bounds(lb, ub)
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mins = min(float(S.get_range(0)[0]) for S in out0)
    maxs = max(float(S.get_range(0)[1]) for S in out0)
    tau = mins + (maxs - mins) * (0.25 + 0.5 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    res = verify_refine_input(S_in, net, spec, bound_mode=bound_mode,
                              heuristic=heuristic, node_budget=200000, min_width=1e-7)
    assert res.status != Status.UNKNOWN, f"ran out of budget (seed={seed})"
    assert res.status == oracle, f"{heuristic}/{bound_mode}: {res.status} != {oracle} (seed={seed})"
    if res.status == Status.SAT:
        assert res.counterexample_x is not None
        with torch.no_grad():
            y = net(torch.as_tensor(res.counterexample_x, dtype=torch.float64))
        assert spec.is_unsafe(y.numpy().flatten()), "returned a fake CE"


@pytest.mark.parametrize("seed", range(6))
def test_input_split_no_false_unsat_vs_sampling(seed):
    rng = np.random.default_rng(2000 + seed)
    net = _rand_net(seed + 50)
    lb = rng.uniform(-1.0, -0.2, size=3); ub = rng.uniform(0.2, 1.0, size=3)
    S_in = Star.from_bounds(lb, ub)
    X = rng.uniform(lb, ub, size=(40000, 3))
    with torch.no_grad():
        Y = net(torch.as_tensor(X, dtype=torch.float64)).numpy()
    tau = float(np.quantile(Y[:, 0], 0.10))
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
    assert np.any(Y[:, 0] <= tau)
    res = verify_refine_input(S_in, net, spec, node_budget=200000, min_width=1e-7)
    assert res.status == Status.SAT, f"false non-SAT ({res.status}) despite sampled CE (seed={seed})"
