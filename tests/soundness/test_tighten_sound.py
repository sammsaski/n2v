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
    from n2v.utils.lpsolver_gpu import gpu_available
    if gpu_available():
        modes.append("dual_gpu")
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


@pytest.mark.parametrize("seed", range(4))
def test_prebuilt_root_is_equivalent(seed):
    """verify_refine(root=...) must reproduce the fresh-root run exactly
    (verdict AND node count) -- the root is spec-independent and only read."""
    rng = np.random.default_rng(200 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    layers = extract_layers(net)
    for mode in ("box", "lp_cpu"):
        fresh = verify_refine(S_in, net, spec, FaithfulSelector(),
                              layers=layers, bound_mode=mode, node_budget=20000)
        root, _ = relaxed_reach(S_in, layers, {}, bound_mode=mode)
        # two disjunct-style reuses of the same root
        for _ in range(2):
            reused = verify_refine(S_in, net, spec, FaithfulSelector(),
                                   layers=layers, bound_mode=mode,
                                   node_budget=20000, root=root)
            assert reused.status == fresh.status
            assert reused.nodes == fresh.nodes

    with pytest.raises(ValueError):
        root, _ = relaxed_reach(S_in, layers, {}, bound_mode="box")
        verify_refine(S_in, net, spec, FaithfulSelector(), layers=layers,
                      bound_mode="lp_cpu", root=root)


gpu = pytest.mark.skipif(
    not __import__("n2v.utils.lpsolver_gpu", fromlist=["gpu_available"]).gpu_available(),
    reason="no CUDA",
)


@gpu
@pytest.mark.parametrize("seed", range(10))
def test_certified_prune_implies_exact_unsat(seed):
    """certified_prune=True must imply the exact disjointness LP also proves
    emptiness (the NS bound is outward, so a GPU prune can never be a false
    prune). The converse may fail (fewer prunes) -- that is allowed."""
    from n2v.refine.witness import certified_prune
    from n2v.sets.halfspace import HalfSpace
    from n2v.utils.verify_specification import verify_specification

    rng = np.random.default_rng(300 + seed)
    net = _rand_net(seed, dims=(2, 6, 6, 2))
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
    lo, hi = root.get_ranges()
    # sweep thresholds across (and beyond) the reachable range of output 0 to
    # exercise both prunable and non-prunable specs
    for frac in (-0.2, 0.1, 0.5, 0.9, 1.2):
        tau = float(lo[0] + (hi[0] - lo[0]) * frac)
        spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
        if certified_prune(root, spec):
            exact = verify_specification([root], HalfSpace(spec.G, spec.g)).verdict
            assert exact == "UNSAT", f"false GPU prune at tau={tau}"


@gpu
@pytest.mark.parametrize("seed", range(6))
def test_verdict_invariant_with_gpu_prune(seed):
    """(bound_mode=dual_gpu, prune_mode=dual_gpu) must reproduce the oracle
    verdict -- the GPU prune changes node count, never the answer."""
    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    res = verify_refine(S_in, net, spec, FaithfulSelector(),
                        bound_mode="dual_gpu", prune_mode="dual_gpu",
                        node_budget=20000)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"gpu-prune verdict {res.status} != {oracle}"


@gpu
@pytest.mark.parametrize("seed", range(6))
def test_verdict_invariant_with_gpu_witness_and_prune(seed):
    """The full GPU node path (dual_gpu bounds + GPU prune + approximate GPU
    witness) must reproduce the oracle verdict. The witness is approximate --
    SAT soundness rests on the exact forward-pass CE check, UNSAT on the
    certified prune / exact pruning of children -- so only node count may
    change, never the answer."""
    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    res = verify_refine(S_in, net, spec, FaithfulSelector(),
                        bound_mode="dual_gpu", prune_mode="dual_gpu",
                        witness_mode="dual_gpu", node_budget=20000)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"gpu-witness verdict {res.status} != {oracle}"


@gpu
@pytest.mark.parametrize("seed", range(4))
def test_population_bounds_enclose_exact(seed):
    """neuron_bounds_population (the frontier/sibling batching primitive) must
    be outward of the exact per-star LP range for every star in the batch,
    including heterogeneous shapes and dual warm-starts."""
    from n2v.refine.tighten import neuron_bounds, neuron_bounds_population

    rng = np.random.default_rng(400 + seed)
    stars = []
    for j, dims in enumerate([(2, 6, 6, 2), (2, 8, 8, 2)]):
        net = _rand_net(seed * 7 + j, dims=dims)
        S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
        stars.append(root)

    # cold and (arbitrary-valued, hence still-sound) warm-started runs
    warm = [{d: (np.abs(rng.normal(size=S.C.shape[0])).astype(np.float32),) * 2
             for d in range(S.dim)} for S in stars]
    for w in (None, warm):
        res = neuron_bounds_population(stars, warm=w)
        for S, (lb_p, ub_p, duals) in zip(stars, res):
            lb_e, ub_e = neuron_bounds(S, "lp_cpu")
            assert np.all(lb_p <= lb_e + 1e-5), "population lb not outward"
            assert np.all(ub_p >= ub_e - 1e-5), "population ub not outward"


@gpu
@pytest.mark.parametrize("seed", range(6))
def test_verdict_invariant_full_gpu_wave(seed):
    """The complete scaled node path -- dual_gpu bounds + sibling-pair lockstep
    reach + wave prune (frontier_width > 1) + GPU witness + dual inheritance --
    must reproduce the oracle verdict."""
    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    res = verify_refine(S_in, net, spec, FaithfulSelector(),
                        bound_mode="dual_gpu", prune_mode="dual_gpu",
                        witness_mode="dual_gpu", frontier_width=4,
                        node_budget=20000)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"wave verdict {res.status} != {oracle}"
