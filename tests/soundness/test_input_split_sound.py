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
from n2v.refine.input_split import (
    verify_refine_input, verify_refine_input_parallel, verify_refine_input_sb,
)
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
@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu", "zono", "zono_lp"])
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


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu", "zono_lp"])
@pytest.mark.parametrize("seed", range(6))
def test_parallel_frontier_matches_oracle(seed, bound_mode):
    """The parallel-frontier driver yields the same verdict as the exact oracle
    (and hence the serial driver) -- threads only change *how* the independent
    sub-problems are scheduled, never the answer."""
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
    res = verify_refine_input_parallel(S_in, net, spec, bound_mode=bound_mode,
                                       node_budget=200000, min_width=1e-7,
                                       n_workers=4, batch=8)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"{bound_mode}: {res.status} != {oracle} (seed={seed})"
    if res.status == Status.SAT:
        with torch.no_grad():
            y = net(torch.as_tensor(res.counterexample_x, dtype=torch.float64))
        assert spec.is_unsafe(y.numpy().flatten())


@pytest.mark.skipif(
    not __import__("n2v.utils.lpsolver_gpu", fromlist=["gpu_available"]).gpu_available(),
    reason="no CUDA",
)
@pytest.mark.parametrize("seed", range(6))
def test_input_split_gpu_matches_oracle(seed):
    """GPU frontier-batched OBBT yields the same verdict as the exact oracle: the
    NS-certified PDHG bounds are a sound outward enclosure (intersected with the
    box), so the relaxation stays sound -- only node counts change with tightness."""
    from n2v.refine.input_split_gpu import verify_refine_input_gpu
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
    res = verify_refine_input_gpu(S_in, net, spec, node_budget=100000, min_width=1e-7,
                                  batch=16, gpu_iters=3000)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"gpu: {res.status} != {oracle} (seed={seed})"
    if res.status == Status.SAT:
        with torch.no_grad():
            y = net(torch.as_tensor(res.counterexample_x, dtype=torch.float64))
        assert spec.is_unsafe(y.numpy().flatten())


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu"])
@pytest.mark.parametrize("seed", range(6))
def test_input_split_sb_matches_oracle(seed, bound_mode):
    """Exact input strong branching yields the same verdict as the oracle -- it
    only changes *which* dimension is split (and how few nodes), never the answer."""
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
    res = verify_refine_input_sb(S_in, net, spec, bound_mode=bound_mode,
                                 node_budget=200000, min_width=1e-7, n_workers=4)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"{bound_mode}: {res.status} != {oracle} (seed={seed})"
    if res.status == Status.SAT:
        with torch.no_grad():
            y = net(torch.as_tensor(res.counterexample_x, dtype=torch.float64))
        assert spec.is_unsafe(y.numpy().flatten())


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


class TestInputCrown:
    """Soundness of the batched-CROWN input-split driver."""

    @staticmethod
    def _setup(seed, dims=(2, 6, 6, 2)):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        mods = []
        for i in range(len(dims) - 2):
            mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        mods += [nn.Linear(dims[-2], dims[-1])]
        return nn.Sequential(*mods).double()

    def test_crown_bounds_enclose_exact_outputs(self):
        import torch
        from n2v.refine.crown_bounds import CrownBatch, prepare_layers
        from n2v.refine.reach_relaxed import extract_layers
        rng = np.random.default_rng(0)
        net = self._setup(3)
        layers = extract_layers(net)
        LB = rng.uniform(-1, -0.1, size=(16, 2))
        UB = LB + rng.uniform(0.2, 1.0, size=(16, 2))
        cb = CrownBatch(prepare_layers(layers), LB, UB)
        Al, bl, Au, bu = cb.output_forms()
        for b in range(16):
            X = rng.uniform(LB[b], UB[b], size=(200, 2))
            with torch.no_grad():
                Y = net(torch.as_tensor(X)).numpy()
            lo = (Al[b].numpy() @ X.T).T + bl[b].numpy()
            hi = (Au[b].numpy() @ X.T).T + bu[b].numpy()
            assert np.all(Y >= lo - 1e-9), "CROWN lower form violated"
            assert np.all(Y <= hi + 1e-9), "CROWN upper form violated"

    @pytest.mark.parametrize("seed", range(8))
    def test_verdict_matches_oracle(self, seed):
        from n2v.refine import verify_refine_input_crown
        rng = np.random.default_rng(100 + seed)
        net = self._setup(seed)
        S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        out0 = reach_pytorch_model(net, S_in, method="exact")
        mn = min(float(S.get_range(0)[0]) for S in out0)
        mx = max(float(S.get_range(0)[1]) for S in out0)
        tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
        spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
        oracle = _oracle(net, S_in, spec)
        # node-budget bounded (deterministic), no wall-clock coupling: tiny
        # nets decide within a few hundred boxes
        res = verify_refine_input_crown(S_in, net, spec, node_budget=200_000)
        assert res.status != Status.UNKNOWN
        assert res.status == oracle, f"crown verdict {res.status} != {oracle}"

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("row_rule", ["argmax", "argmin"])
    def test_verdict_matches_oracle_multirow(self, seed, row_rule):
        """Multi-row unsafe regions exercise the driving-row selection (the
        single-row tests cannot: argmin == argmax there)."""
        from n2v.refine import verify_refine_input_crown
        rng = np.random.default_rng(700 + seed)
        net = self._setup(seed)
        S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        out0 = reach_pytorch_model(net, S_in, method="exact")
        mn0 = min(float(x.get_range(0)[0]) for x in out0)
        mx0 = max(float(x.get_range(0)[1]) for x in out0)
        mn1 = min(float(x.get_range(1)[0]) for x in out0)
        mx1 = max(float(x.get_range(1)[1]) for x in out0)
        t0 = mn0 + (mx0 - mn0) * (0.3 + 0.4 * rng.random())
        t1 = mn1 + (mx1 - mn1) * (0.3 + 0.4 * rng.random())
        # unsafe = {y0 <= t0 AND y1 <= t1}: G = -I rows (y_i >= -t_i form)
        spec = LinearSpec(G=np.array([[1.0, 0.0], [0.0, 1.0]]),
                          g=np.array([t0, t1]))
        oracle = _oracle(net, S_in, spec)
        res = verify_refine_input_crown(S_in, net, spec, node_budget=200_000,
                                        row_rule=row_rule)
        assert res.status != Status.UNKNOWN
        assert res.status == oracle, f"multirow {row_rule}: {res.status} != {oracle}"

    @pytest.mark.parametrize("seed", range(4))
    def test_constrained_input_star_stays_sound(self, seed):
        """On a genuinely constrained input star (extra C/d halfspace) the
        driver searches the BOUNDING box: it must never report a SAT whose
        counterexample lies outside the true input set, and must never report
        UNSAT when the exact oracle (which respects C/d) finds the set unsafe.
        UNKNOWN is acceptable (documented incompleteness for polytope inputs)."""
        from n2v.refine import verify_refine_input_crown
        rng = np.random.default_rng(500 + seed)
        net = self._setup(seed)
        S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        S2 = Star(S.V, np.vstack([S.C, [[1.0, 1.0]]]),
                  np.vstack([S.d, [[0.5]]]), S.predicate_lb, S.predicate_ub)
        out0 = reach_pytorch_model(net, S2, method="exact")
        mn = min(float(x.get_range(0)[0]) for x in out0)
        mx = max(float(x.get_range(0)[1]) for x in out0)
        tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
        spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
        oracle = _oracle(net, S2, spec)
        res = verify_refine_input_crown(S2, net, spec, time_budget=20.0)
        if res.status == Status.SAT:
            assert oracle == Status.SAT
            assert np.all(S2.contains(
                np.asarray(res.counterexample_x).reshape(1, -1), method="lp"))
        elif res.status == Status.UNSAT:
            assert oracle == Status.UNSAT

    def test_out_of_set_unsafe_vertex_is_never_sat(self):
        """Deterministic non-vacuous membership check: identity net, unsafe
        region = exactly the box corner that a C/d halfspace CUTS OFF. The
        CROWN driver searches the bounding box, so its candidate vertex IS
        that corner and its exact forward IS unsafe -- the zero-tolerance
        membership filter is the only thing standing between this and a false
        SAT. The box also can't be pruned (the unsafe corner is in the
        bounding box), so a SAT return here is a soundness bug, full stop."""
        import torch.nn as nn
        import torch as th
        from n2v.refine import verify_refine_input_crown
        net = nn.Sequential(nn.Linear(2, 2))
        with th.no_grad():
            net[0].weight.copy_(th.eye(2))
            net[0].bias.zero_()
        net = net.double()
        S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        # cut off the (1,1) corner: alpha_0 + alpha_1 <= 0.5
        S2 = Star(S.V, np.vstack([S.C, [[1.0, 1.0]]]),
                  np.vstack([S.d, [[0.5]]]), S.predicate_lb, S.predicate_ub)
        # unsafe = {y_0 >= 0.9 AND y_1 >= 0.9} -- reachable only from the cut
        # corner: G = -I, g = -0.9
        spec = LinearSpec(G=-np.eye(2), g=np.array([-0.9, -0.9]))
        res = verify_refine_input_crown(S2, net, spec, node_budget=20_000)
        assert res.status != Status.SAT, \
            "SAT with an out-of-set counterexample (membership filter broken)"
