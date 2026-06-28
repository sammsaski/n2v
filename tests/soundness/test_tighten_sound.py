"""
Soundness of LP-over-P bound tightening.

Tightening changes which neurons are relaxed/split -- it must NOT change the
verdict (only node count) and must keep the reach a sound over-approximation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.star import Star
from n2v.utils.lpsolver import check_feasibility
from n2v.refine import FaithfulSelector, LinearSpec, Status, verify_refine
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach


def _rand_net(seed, dims=(2, 5, 5, 2)):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers).double()


def _intersects_unsafe(S, spec):
    A = [spec.G @ S.V[:, 1:]]
    b = [spec.g - spec.G @ S.V[:, 0]]
    if S.C.size > 0:
        A.append(S.C)
        b.append(S.d.flatten())
    return check_feasibility(A=np.vstack(A), b=np.concatenate(b),
                             lb=S.predicate_lb.flatten(), ub=S.predicate_ub.flatten())


def _oracle(model, S_in, spec):
    for S in reach_pytorch_model(model, S_in, method="exact"):
        if _intersects_unsafe(S, spec):
            return Status.SAT
    return Status.UNSAT


@pytest.mark.parametrize("seed", range(6))
def test_lp_reach_is_sound_and_tighter_than_box(seed):
    rng = np.random.default_rng(seed)
    net = _rand_net(seed, dims=(3, 6, 6, 2))
    layers = extract_layers(net)
    lb = rng.uniform(-1, -0.2, size=3)
    ub = rng.uniform(0.2, 1, size=3)
    S_in = Star.from_bounds(lb, ub)

    S_box, _ = relaxed_reach(S_in, layers, bound_mode="box")
    S_lp, _ = relaxed_reach(S_in, layers, bound_mode="lp_cpu")

    # sound: exact output samples inside the LP-tightened reach
    X = rng.uniform(lb, ub, size=(400, 3))
    with torch.no_grad():
        Y = net(torch.as_tensor(X, dtype=torch.float64)).numpy()
    assert np.all(S_lp.contains(Y, method="lp")), "LP-tightened reach is unsound"

    # tighter: the LP reach's output bounding box is within the box reach's
    lo_box, hi_box = S_box.get_ranges()
    lo_lp, hi_lp = S_lp.get_ranges()
    assert np.all(lo_lp >= lo_box - 1e-6)
    assert np.all(hi_lp <= hi_box + 1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_verdict_invariant_across_bound_modes(seed):
    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    modes = ["box", "lp_cpu"]
    for mode in modes:
        res = verify_refine(S_in, net, spec, FaithfulSelector(),
                            bound_mode=mode, node_budget=20000)
        assert res.status != Status.UNKNOWN
        assert res.status == oracle, f"bound_mode={mode}: {res.status} != oracle {oracle}"


def test_lp_tightening_stabilises_neurons_and_cuts_nodes():
    """LP bound-tightening should yield <= the box node count on a net where C/d bites."""
    rng = np.random.default_rng(3)
    net = _rand_net(7, dims=(2, 8, 8, 2))
    S_in = Star.from_bounds(np.array([-0.8, -0.8]), np.array([0.8, 0.8]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    spec = LinearSpec(G=np.array([[1.0, -1.0]]), g=np.array([mn - 0.3]))  # hard/UNSAT

    box = verify_refine(S_in, net, spec, FaithfulSelector(), bound_mode="box",
                        node_budget=20000)
    lp = verify_refine(S_in, net, spec, FaithfulSelector(), bound_mode="lp_cpu",
                       node_budget=20000)
    assert box.status == lp.status  # same verdict
    # tighter bounds never increase the node count for the same selector
    assert lp.nodes <= box.nodes
