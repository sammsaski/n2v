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
from tests.unit.refine.conftest import _assert_honest_ce


# The exact CPU configuration, pinned so a CUDA host does not switch these
# tests to the GPU defaults.
CPU_CONFIG = dict(bound_mode="box", prune_mode="exact", witness_mode="exact",
                  frontier_width=1)

# Spec ``y0 <= tau`` placed relative to the exact range [mn, mx] of output 0:
# inside the range (SAT: the image of the box is connected), below it, and
# just below it (both UNSAT).
_TAU_ARMS = {"inside": (0.5, Status.SAT), "below": (-0.1, Status.UNSAT),
             "near": (-0.01, Status.UNSAT)}


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


def _tau_arm_spec(model, S_in, arm):
    """The tau-arm spec on output 0 and its ground truth."""
    out0 = reach_pytorch_model(model, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    frac, truth = _TAU_ARMS[arm]
    spec = LinearSpec(G=np.array([[1.0, 0.0]]),
                      g=np.array([mn + frac * (mx - mn)]))
    return spec, truth


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
    # dual_gpu is never dropped: without CUDA the certified engine runs on CPU
    # torch, which is exactly the path a CPU-only CI must exercise.
    for mode in ("box", "lp_cpu", "dual_gpu"):
        res = verify_refine(S_in, net, spec, FaithfulSelector(),
                            bound_mode=mode, prune_mode="exact",
                            witness_mode="exact", frontier_width=1,
                            node_budget=20000)
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

    box = verify_refine(S_in, net, spec, FaithfulSelector(),
                        node_budget=20000, **CPU_CONFIG)
    lp = verify_refine(S_in, net, spec, FaithfulSelector(), node_budget=20000,
                       **{**CPU_CONFIG, "bound_mode": "lp_cpu"})
    assert box.status == lp.status  # same verdict
    # tighter bounds never increase the node count for the same selector
    assert lp.nodes <= box.nodes


# The certified GPU engine guards below take the ``engine_device`` fixture
# (tests/conftest.py): CPU torch on every host, CUDA where it exists. They
# used to sit behind a CUDA-only skip, so a CPU-only CI never ran the
# production-default engine's soundness checks.


@pytest.mark.parametrize("seed", range(10))
def test_certified_prune_implies_exact_unsat(seed, engine_device):
    """A certified GPU prune must imply the exact disjointness LP also proves
    emptiness (the NS bound is outward, so a GPU prune can never be a false
    prune). The converse may fail (fewer prunes) -- that is allowed."""
    from n2v.refine.witness import certified_prune_wave
    from n2v.sets.halfspace import HalfSpace
    from n2v.utils.verify_specification import verify_specification

    net = _rand_net(seed, dims=(2, 6, 6, 2))
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
    lo, hi = root.get_ranges()
    # sweep thresholds across (and beyond) the reachable range of output 0 to
    # exercise both prunable and non-prunable specs
    for frac in (-0.2, 0.1, 0.5, 0.9, 1.2):
        tau = (lo[0] + (hi[0] - lo[0]) * frac).item()
        spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
        if certified_prune_wave([root], spec)[0]:
            exact = verify_specification([root], HalfSpace(spec.G, spec.g)).verdict
            assert exact == "UNSAT", f"false GPU prune at tau={tau}"


@pytest.mark.parametrize("seed", range(6))
def test_verdict_invariant_with_gpu_prune(seed, engine_device):
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
                        witness_mode="exact", frontier_width=1,
                        node_budget=20000)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"gpu-prune verdict {res.status} != {oracle}"


@pytest.mark.parametrize("seed", range(6))
def test_verdict_invariant_with_gpu_witness_and_prune(seed, engine_device):
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
                        witness_mode="dual_gpu", frontier_width=1,
                        node_budget=20000)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"gpu-witness verdict {res.status} != {oracle}"


@pytest.mark.parametrize("seed", range(4))
def test_population_bounds_enclose_exact(seed, engine_device):
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


@pytest.mark.parametrize("arm", list(_TAU_ARMS))
@pytest.mark.parametrize("seed", range(6))
def test_verdict_invariant_full_gpu_wave(seed, arm, engine_device):
    """The complete scaled node path -- dual_gpu bounds + sibling-pair lockstep
    reach + wave prune (frontier_width > 1) + GPU witness + dual inheritance --
    must reproduce the oracle verdict, on SAT and UNSAT ground truths, with
    honest counterexamples."""
    net = _rand_net(seed)
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    S_in = Star.from_bounds(lb, ub)
    spec, truth = _tau_arm_spec(net, S_in, arm)
    assert _oracle(net, S_in, spec) == truth, "test setup: oracle disagrees"
    res = verify_refine(S_in, net, spec, FaithfulSelector(),
                        bound_mode="dual_gpu", prune_mode="dual_gpu",
                        witness_mode="dual_gpu", frontier_width=4,
                        node_budget=20000)
    assert res.status == truth, f"wave verdict {arm}: {res.status} != {truth}"
    _assert_honest_ce(res, net, spec, lb, ub)


@pytest.mark.parametrize("arm", list(_TAU_ARMS))
@pytest.mark.parametrize("seed", range(6))
def test_default_configuration_matches_oracle(seed, arm, monkeypatch, engine_device):
    """With CUDA, the defaults are the certified GPU configuration (dual_gpu
    bounds, prune and witnesses over waves of 32 nodes): every arm must be
    decided (never UNKNOWN) and match the ground truth, with honest
    counterexamples. The certified GPU prune must actually run (the CPU
    configuration gives the same verdicts)."""
    import n2v.refine.bab as bab

    prunes = []
    real_prune = bab.certified_prune_wave

    def counting_prune(*args, **kwargs):
        prunes.append(1)
        return real_prune(*args, **kwargs)

    monkeypatch.setattr(bab, "certified_prune_wave", counting_prune)
    if engine_device == "cpu":
        # The defaults are chosen from ``bab.gpu_available`` (its own import
        # of the name); the fixture only redirects the device resolution to
        # CPU torch. Present a CUDA host here so the GPU configuration is
        # selected and then runs on CPU torch.
        monkeypatch.setattr(bab, "gpu_available", lambda: True)
    net = _rand_net(seed)
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    S_in = Star.from_bounds(lb, ub)
    spec, truth = _tau_arm_spec(net, S_in, arm)
    res = verify_refine(S_in, net, spec, node_budget=20000)
    assert prunes, "the defaults did not run the certified GPU prune"
    assert res.status == truth, \
        f"default configuration {arm}: {res.status} != {truth}"
    _assert_honest_ce(res, net, spec, lb, ub)


@pytest.mark.parametrize("seed", range(4))
def test_layer_order_selector_verdict(seed):
    """LayerOrderSelector (the enumeration-regime branch order) is just another
    selector: exhaustive splitting stays exhaustive, so verdicts match the
    oracle -- only the tree shape changes."""
    from n2v.refine import LayerOrderSelector

    rng = np.random.default_rng(500 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    res = verify_refine(S_in, net, spec, LayerOrderSelector(),
                        node_budget=20000, **CPU_CONFIG)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"layer-order verdict {res.status} != {oracle}"


@pytest.mark.parametrize("seed", range(5))
def test_contract_input_boxes_outward_of_exact(seed, engine_device):
    """Certified domain contraction must stay OUTWARD of the exact input-dim
    range over {C alpha <= d, box} (never cut a feasible point), and must
    declare ``empty`` only for a truly infeasible system."""
    from n2v.utils.lp_solver_enum import LPSolver
    from n2v.refine.tighten import contract_input_boxes

    net = _rand_net(600 + seed, dims=(3, 6, 6, 2))
    S_in = Star.from_bounds(np.array([-1.0, -0.5, -1.0]),
                            np.array([0.5, 1.0, 1.0]))
    root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
    assert root.C.size > 0
    k0 = S_in.nVar

    (clb, cub, empty), = contract_input_boxes(
        [(root.C, root.d)], [(root.predicate_lb, root.predicate_ub)], k0)

    # exact range of each input dim over P (HiGHS on the alpha-space objectives)
    objectives, flags = [], []
    for i in range(k0):
        e = np.zeros(root.nVar)
        e[i] = 1.0
        objectives.extend([e, e])
        flags.extend([True, False])
    exact = root._solve_lp_batch(objectives, flags, LPSolver.DEFAULT)
    feasible = all(v is not None for v in exact)
    if feasible:
        assert not empty, "certified-empty on a feasible system"
        for i in range(k0):
            assert clb[i, 0] <= exact[2 * i] + 1e-7, "contraction cut the min"
            assert cub[i, 0] >= exact[2 * i + 1] - 1e-7, "contraction cut the max"
    # contraction never widens the box
    assert np.all(clb >= root.predicate_lb[:k0] - 1e-12)
    assert np.all(cub <= root.predicate_ub[:k0] + 1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_witness_wave_matches_scalar_contract(seed, engine_device):
    """The batched wave witness must honour the scalar witness's contract per
    star: alpha box-feasible by construction, and t equal to the EXACT max
    spec-row margin evaluated at alpha."""
    from n2v.refine.witness import violation_witness_wave

    stars = []
    for j in range(3):
        net = _rand_net(1000 + seed * 3 + j, dims=(2, 6, 6, 2))
        S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
        stars.append(root)
    spec = LinearSpec(G=np.array([[1.0, -1.0]]), g=np.array([0.0]))

    for S, res in zip(stars, violation_witness_wave(stars, spec)):
        assert res is not None
        alpha, t = res
        assert np.all(alpha >= S.predicate_lb.flatten() - 1e-6)
        assert np.all(alpha <= S.predicate_ub.flatten() + 1e-6)
        y = S.V[:, 0] + S.V[:, 1:] @ alpha
        assert abs(t - float(np.max(spec.margins(y)))) < 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_verdict_invariant_contract_split_config(seed, engine_device):
    """The contract-split configuration (layer-ordered branching + certified
    input contraction at every split + group reach with certified dual bounds
    at every layer + certified GPU prune) must reproduce the oracle verdict,
    with either witness mode: contraction is NS-outward (never cuts a feasible
    input), dropped children are certified-empty, and SAT still rests on the
    exact forward-pass CE check. The bound_mode="box" arm covers the
    per-child fallback reach path in split_wave (non-dual bound modes)."""
    from n2v.refine import LayerOrderSelector

    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    for wmode, bmode in (("dual_gpu", "dual_gpu"), ("exact", "dual_gpu"),
                         ("dual_gpu", "box")):
        res = verify_refine(S_in, net, spec, LayerOrderSelector(),
                            bound_mode=bmode, contract_inputs=True,
                            prune_mode="dual_gpu", witness_mode=wmode,
                            frontier_width=4, node_budget=20000)
        assert res.status != Status.UNKNOWN
        assert res.status == oracle, \
            f"contract-split verdict ({wmode}, {bmode}) {res.status} != {oracle}"
