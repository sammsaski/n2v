"""
Option handling and search logic of ``verify_refine`` (and the entry checks
it shares with ``verify_refine_input_crown``).

  * Validation: unknown mode strings (including the removed bound modes), a
    non-positive ``frontier_width``, a multi-disjunct spec combined with
    ``contract_inputs`` and a non-float64 model are rejected.
  * Default resolution: the GPU configuration when CUDA is usable, the CPU
    configuration otherwise, each knob independently; both match the exact
    oracle, and the GPU configuration also runs on CPU torch.
  * Multi-disjunct specs: witness and split guidance follow the disjunct with
    the smallest worst-row margin, and verdicts match an exact-reach oracle.
  * LP failures are never a certificate: a failed disjointness LP inside
    ``verify_specification`` never yields UNSAT, and a failed wave-witness LP
    leaves its nodes undecided (UNKNOWN).
  * Entry checks of both drivers: an empty-input report gives UNSAT only when
    the emptiness is confirmed; counterexample checks run in eval mode
    without touching the caller's model.
  * The low-input-dimension contraction stack runs on CPU torch.
"""

import functools

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.lpsolver import LPFailure, check_feasibility
from n2v.refine import (FaithfulSelector, LayerOrderSelector, LinearSpec,
                        Status, verify_refine, verify_refine_input_crown)
from n2v.refine.bab import _eval_mode_model, _most_violated, _resolve_defaults
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.witness import violation_lp
from tests.unit.refine.conftest import _assert_honest_ce, seq_net


CPU_CONFIG = dict(bound_mode="box", prune_mode="exact", witness_mode="exact",
                  frontier_width=1)
GPU_CONFIG = dict(bound_mode="dual_gpu", prune_mode="dual_gpu",
                  witness_mode="dual_gpu", frontier_width=4)

# A spec every output satisfies: any honest run finds a counterexample (SAT).
spec_everything = LinearSpec([[1.0, 0.0]], [1e9])

# CUDA-only arms: the ``gpu`` tier marker plus a skip without a device. The
# certified engine itself is exercised on CPU torch by the "gpu-on-cpu-torch"
# arms and the ``engine_device`` fixture, never behind this marker.
gpu = [pytest.mark.gpu, pytest.mark.skipif(
    not __import__("n2v.utils.lpsolver_gpu", fromlist=["gpu_available"]).gpu_available(),
    reason="no CUDA",
)]


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


def _box():
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    return lb, ub, Star.from_bounds(lb, ub)


def _raise_lp_failure(*args, **kwargs):
    raise LPFailure("simulated")


def _gpu_engine_on_cpu_torch(monkeypatch):
    """Run the certified GPU engine on CPU torch: device ``None`` resolves to
    "cpu" (the path CI takes)."""
    monkeypatch.setattr("n2v.utils.lpsolver_gpu.gpu_available", lambda: False)


def _spy_engine(monkeypatch):
    """Record which engine a ``verify_refine`` run used: the ``bound_mode`` of
    its root reach and the number of certified GPU prune calls. Verdicts alone
    cannot tell the configurations apart (both match the oracle)."""
    import n2v.refine.bab as bab

    seen = {"bound_modes": [], "prunes": 0}
    real_reach, real_prune = bab.relaxed_reach, bab.certified_prune_wave

    def reach(*args, **kwargs):
        seen["bound_modes"].append(kwargs.get("bound_mode"))
        return real_reach(*args, **kwargs)

    def prune(*args, **kwargs):
        seen["prunes"] += 1
        return real_prune(*args, **kwargs)

    monkeypatch.setattr(bab, "relaxed_reach", reach)
    monkeypatch.setattr(bab, "certified_prune_wave", prune)
    return seen


# Spec ``y0 <= tau`` placed relative to the exact range [mn, mx] of output 0:
# inside the range (SAT: the image of the box is connected), below it, and
# just below it (both UNSAT).
_TAU_ARMS = {"inside": (0.5, Status.SAT), "below": (-0.1, Status.UNSAT),
             "near": (-0.01, Status.UNSAT)}


@functools.lru_cache(maxsize=None)
def _output0_range(seed):
    _, _, S_in = _box()
    out0 = reach_pytorch_model(seq_net(2, 5, 5, 2, seed=seed), S_in,
                               method="exact")
    return (min(float(S.get_range(0)[0]) for S in out0),
            max(float(S.get_range(0)[1]) for S in out0))


def _tau_instance(seed, arm):
    """``seq_net(2, 5, 5, 2)`` over [-1, 1]^2 with the tau-arm spec and its
    ground truth."""
    mn, mx = _output0_range(seed)
    frac, truth = _TAU_ARMS[arm]
    spec = LinearSpec([[1.0, 0.0]], [mn + frac * (mx - mn)])
    lb, ub, S_in = _box()
    return seq_net(2, 5, 5, 2, seed=seed), lb, ub, S_in, spec, truth


def _run_driver(driver, S_in, net, spec, **kwargs):
    if driver == "verify_refine":
        return verify_refine(S_in, net, spec, FaithfulSelector(),
                             **{**CPU_CONFIG, **kwargs})
    return verify_refine_input_crown(S_in, net, spec, node_budget=5000,
                                     **kwargs)


_DRIVERS = ["verify_refine", "verify_refine_input_crown"]


# --------------------------------------------------------------------------- #
# Option validation                                                            #
# --------------------------------------------------------------------------- #
def test_unknown_mode_strings_rejected():
    """Typo'd prune/witness modes must raise, not silently run a hybrid."""
    net = seq_net(2, 5, 5, 2, seed=0)
    _, _, S_in = _box()
    with pytest.raises(ValueError, match="prune_mode"):
        verify_refine(S_in, net, spec_everything,
                      **{**CPU_CONFIG, "prune_mode": "gpu_dual"})
    with pytest.raises(ValueError, match="witness_mode"):
        verify_refine(S_in, net, spec_everything,
                      **{**CPU_CONFIG, "witness_mode": "center"})


def test_removed_bound_modes_rejected():
    """Bound modes removed from the refine engine raise instead of running."""
    net = seq_net(2, 5, 5, 2, seed=0)
    _, _, S_in = _box()
    for mode in ("zono", "zono_lp", "lp_gpu"):
        with pytest.raises(ValueError, match="bound_mode"):
            verify_refine(S_in, net, spec_everything,
                          **{**CPU_CONFIG, "bound_mode": mode})


def test_frontier_width_must_be_positive():
    net = seq_net(2, 5, 5, 2, seed=0)
    _, _, S_in = _box()
    with pytest.raises(ValueError, match="frontier_width"):
        verify_refine(S_in, net, spec_everything,
                      **{**CPU_CONFIG, "frontier_width": 0})


@pytest.mark.parametrize("driver", _DRIVERS)
def test_non_float64_model_rejected(driver):
    """A float32 model would crash at the first SAT candidate (the checks feed
    float64 inputs), so both drivers reject it at entry."""
    net = seq_net(2, 5, 5, 2, seed=0).float()
    _, _, S_in = _box()
    with pytest.raises(TypeError, match="float64"):
        _run_driver(driver, S_in, net, spec_everything)


def test_multi_disjunct_with_contract_inputs_rejected():
    net = seq_net(2, 5, 5, 2, seed=0)
    _, _, S_in = _box()
    with pytest.raises(TypeError):
        verify_refine(S_in, net, [spec_everything, spec_everything],
                      contract_inputs=True, **CPU_CONFIG)


def test_immediate_deadline_on_a_wave_returns_unknown():
    """An expired deadline returns UNKNOWN under wave processing, even on an
    instance that is otherwise SAT at the first node."""
    net = seq_net(2, 5, 5, 2, seed=0)
    _, _, S_in = _box()
    res = verify_refine(S_in, net, spec_everything, FaithfulSelector(),
                        time_budget=0.0, **{**CPU_CONFIG, "frontier_width": 4})
    assert res.status is Status.UNKNOWN


# --------------------------------------------------------------------------- #
# Default resolution                                                           #
# --------------------------------------------------------------------------- #
def test_defaults_resolve_to_the_host_configuration():
    assert _resolve_defaults(None, None, None, None, cuda=True) == (
        "dual_gpu", "dual_gpu", "dual_gpu", 32)
    assert _resolve_defaults(None, None, None, None, cuda=False) == (
        "box", "exact", "exact", 1)


def test_explicit_knobs_win_independently():
    assert _resolve_defaults("lp_cpu", None, "exact", None, True) == (
        "lp_cpu", "dual_gpu", "exact", 32)


@pytest.mark.parametrize("arm", list(_TAU_ARMS))
def test_defaults_without_cuda_match_oracle(arm, monkeypatch):
    """No CUDA: the defaults are the exact CPU configuration."""
    monkeypatch.setattr("n2v.refine.bab.gpu_available", lambda: False)
    _gpu_engine_on_cpu_torch(monkeypatch)
    seen = _spy_engine(monkeypatch)
    net, lb, ub, S_in, spec, truth = _tau_instance(0, arm)
    res = verify_refine(S_in, net, spec)
    assert seen["bound_modes"] == ["box"] and seen["prunes"] == 0, seen
    assert res.status == truth, f"{arm}: {res.status} != {truth}"
    _assert_honest_ce(res, net, spec, lb, ub)


@pytest.mark.parametrize("arm", list(_TAU_ARMS))
@pytest.mark.parametrize("seed", [0, 1])
def test_default_gpu_configuration_on_cpu_torch(seed, arm, monkeypatch):
    """The defaults resolve to the GPU configuration, whose certified engine
    then runs on CPU torch: verdicts must still match the oracle, with honest
    counterexamples."""
    monkeypatch.setattr("n2v.refine.bab.gpu_available", lambda: True)
    _gpu_engine_on_cpu_torch(monkeypatch)
    seen = _spy_engine(monkeypatch)
    net, lb, ub, S_in, spec, truth = _tau_instance(seed, arm)
    res = verify_refine(S_in, net, spec, gpu_iters=50, node_budget=20000)
    assert seen["bound_modes"] == ["dual_gpu"] and seen["prunes"] >= 1, seen
    assert res.status == truth, f"seed={seed} {arm}: {res.status} != {truth}"
    _assert_honest_ce(res, net, spec, lb, ub)


# --------------------------------------------------------------------------- #
# Multi-disjunct specs                                                         #
# --------------------------------------------------------------------------- #
def test_most_violated_picks_the_smallest_margin():
    a = np.zeros(2)
    cands = [("A", (a, -0.1)), ("B", (a, -2.0)), ("C", (a, 0.5))]
    assert _most_violated(cands)[0] == "B"


def test_guidance_follows_the_most_violated_disjunct(monkeypatch):
    """With two reachable disjuncts, the split witness is built for the one
    whose witness lies deepest inside its unsafe region (smallest ``t``), not
    the least violated one."""
    import n2v.refine.bab as bab

    net = seq_net(2, 5, 5, 2, seed=0)
    _, _, S_in = _box()
    layers = extract_layers(net)
    root, _ = relaxed_reach(S_in, layers, {}, bound_mode="box")
    assert root.relax_meta
    lo, hi = (float(v) for v in root.get_range(0))
    s1 = LinearSpec(np.array([[1.0, 0.0]]), np.array([lo + 0.2 * (hi - lo)]))
    s2 = LinearSpec(np.array([[1.0, 0.0]]), np.array([lo + 0.8 * (hi - lo)]))
    t1 = violation_lp(root, s1, include_Cd=True)[1]
    t2 = violation_lp(root, s2, include_Cd=True)[1]
    expected = s1 if t1 < t2 else s2
    assert expected is s2, "test setup: the deeper threshold must be more violated"

    recorded = []
    real_make_witness = bab.make_witness

    def _recording_make_witness(star, spec, *args, **kwargs):
        recorded.append(spec)
        return real_make_witness(star, spec, *args, **kwargs)

    monkeypatch.setattr(bab, "is_true_counterexample",
                        lambda *a, **k: (False, None, None))
    monkeypatch.setattr(bab, "make_witness", _recording_make_witness)
    verify_refine(S_in, net, [s1, s2], FaithfulSelector(), layers=layers,
                  node_budget=2, **CPU_CONFIG)
    assert recorded, "the root was not split"
    assert recorded[0] is expected


@functools.lru_cache(maxsize=None)
def _disjunct_instance(seed):
    """``seq_net(2, 5, 5, 2)`` over [-1, 1]^2 with two unreachable disjuncts
    (just outside the exact output range) and two reachable ones (at the middle
    of the exact output range)."""
    net = seq_net(2, 5, 5, 2, seed=seed)
    lb, ub, S_in = _box()
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = [min(float(S.get_range(k)[0]) for S in out0) for k in range(2)]
    mx = [max(float(S.get_range(k)[1]) for S in out0) for k in range(2)]
    r = [mx[k] - mn[k] for k in range(2)]
    disjuncts = {
        "below0": LinearSpec([[1.0, 0.0]], [mn[0] - 0.05 * r[0]]),
        "above1": LinearSpec([[0.0, -1.0]], [-(mx[1] + 0.05 * r[1])]),
        "in0": LinearSpec([[1.0, 0.0]], [mn[0] + 0.5 * r[0]]),
        "in1": LinearSpec([[0.0, -1.0]], [-(mn[1] + 0.5 * r[1])]),
    }
    return net, lb, ub, S_in, out0, disjuncts


_DISJUNCT_LISTS = [("below0", "in0"), ("in0", "below0"), ("below0", "above1"),
                   ("in0", "in1"), ("below0", "above1", "in1")]


@pytest.mark.parametrize(
    "config", [pytest.param("cpu", id="cpu"),
               pytest.param("gpu", id="gpu", marks=gpu),
               pytest.param("gpu_on_cpu_torch", id="gpu-on-cpu-torch")])
@pytest.mark.parametrize("names", _DISJUNCT_LISTS,
                         ids=["+".join(n) for n in _DISJUNCT_LISTS])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multi_disjunct_matches_oracle(seed, names, config, monkeypatch):
    """One tree over a disjunction: live-disjunct masks, per-disjunct prunes
    and witnesses, and SAT on any disjunct must reproduce the exact oracle."""
    if config == "gpu_on_cpu_torch":
        _gpu_engine_on_cpu_torch(monkeypatch)
        config = {**GPU_CONFIG, "gpu_iters": 50}
    else:
        config = CPU_CONFIG if config == "cpu" else GPU_CONFIG
    net, lb, ub, S_in, out0, disjuncts = _disjunct_instance(seed)
    specs = [disjuncts[n] for n in names]
    truth = (Status.SAT if any(_star_intersects_unsafe(S, sp)
                               for sp in specs for S in out0)
             else Status.UNSAT)
    assert truth == (Status.UNSAT if set(names) == {"below0", "above1"}
                     else Status.SAT), "test setup: unexpected ground truth"
    res = verify_refine(S_in, net, specs, FaithfulSelector(),
                        node_budget=20000, **config)
    assert res.status == truth, f"{names} seed={seed}: {res.status} != {truth}"
    _assert_honest_ce(res, net, specs, lb, ub)


# --------------------------------------------------------------------------- #
# LP failures are never certificates                                           #
# --------------------------------------------------------------------------- #
def _fail_disjointness_lp(monkeypatch):
    """Make every disjointness LP inside ``verify_specification`` fail with
    ``LPFailure``; returns a list that grows by one entry per call."""
    calls = []

    def _fail(*args, **kwargs):
        calls.append(1)
        raise LPFailure("simulated")

    monkeypatch.setattr(
        "n2v.utils.verify_specification._group_disjoint_from_reach_set", _fail)
    return calls


def test_verify_specification_lp_failure_never_yields_unsat(monkeypatch):
    """``verify_refine`` relies on ``verify_specification`` degrading a failed
    disjointness LP to UNKNOWN (never UNSAT, never an exception) for its exact
    prune."""
    calls = _fail_disjointness_lp(monkeypatch)
    _, _, S_in = _box()
    res = verify_refine(S_in, seq_net(2, 6, 6, 2, seed=5), spec_everything,
                        FaithfulSelector(), node_budget=200, **CPU_CONFIG)
    assert calls, "the exact prune never called verify_specification"
    assert res.status != Status.UNSAT


def test_exact_leaf_backstop_lp_failure_never_yields_unsat(monkeypatch, engine_device):
    """Under the GPU prune and witness, ``verify_specification`` runs only in
    the exact-leaf backstop. Here the root is already an exact star (both
    hidden pre-activations lie in [1, 3]), the GPU prune is patched to never
    prune and the counterexample check to never succeed, so the backstop's
    ``verify_specification`` call is the only step that could prune the root:
    a failed LP there must leave it undecided, not UNSAT."""
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.eye(2, dtype=torch.float64))
        net[0].bias.fill_(2.0)
        net[2].weight.copy_(torch.eye(2, dtype=torch.float64))
        net[2].bias.zero_()
    _, _, S_in = _box()
    layers = extract_layers(net)
    root, _ = relaxed_reach(S_in, layers, {},
                            bound_mode=GPU_CONFIG["bound_mode"])
    assert not root.relax_meta, "test setup: the root must be an exact star"

    monkeypatch.setattr("n2v.refine.bab.certified_prune_wave",
                        lambda stars, *a, **k: [False] * len(stars))
    monkeypatch.setattr("n2v.refine.bab.is_true_counterexample",
                        lambda *a, **k: (False, None, None))
    calls = _fail_disjointness_lp(monkeypatch)
    res = verify_refine(S_in, net, spec_everything, FaithfulSelector(),
                        layers=layers, node_budget=200, **GPU_CONFIG)
    assert calls, "the exact-leaf backstop was not reached"
    assert res.status != Status.UNSAT


def test_wave_witness_lp_failure_is_unknown(monkeypatch):
    """A wave witness whose exact-LP fallback fails leaves its nodes undecided:
    the run returns UNKNOWN instead of raising or pruning them."""
    monkeypatch.setattr("n2v.refine.bab.violation_witness_wave",
                        _raise_lp_failure)
    net = seq_net(2, 5, 5, 2, seed=0)
    _, _, S_in = _box()
    res = verify_refine(S_in, net, spec_everything, FaithfulSelector(),
                        bound_mode="box", prune_mode="exact",
                        witness_mode="dual_gpu", frontier_width=1,
                        node_budget=50)
    assert res.status is Status.UNKNOWN


# --------------------------------------------------------------------------- #
# Entry checks shared by both drivers                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ["unconfirmed", "confirmed",
                                 "confirmation_raises"])
@pytest.mark.parametrize("driver", _DRIVERS)
def test_unconfirmed_empty_input_is_unknown(driver, arm, monkeypatch):
    """The finitizer's emptiness report is acted on only when
    ``Star.is_empty_set`` confirms it (UNSAT at no cost); otherwise the run is
    UNKNOWN, never a vacuous UNSAT. The input set here is feasible, so the
    unconfirmed report is exactly the false-empty case."""
    monkeypatch.setattr("n2v.refine.bab.ensure_finite_predicate_box",
                        lambda star, lp_solver=None: (star, True))
    if arm == "confirmed":
        monkeypatch.setattr(Star, "is_empty_set",
                            lambda self, lp_solver=None: True)
    elif arm == "confirmation_raises":
        monkeypatch.setattr(Star, "is_empty_set", _raise_lp_failure)
    _, _, S_in = _box()
    res = _run_driver(driver, S_in, seq_net(2, 5, 5, 2, seed=0),
                      spec_everything)
    if arm == "confirmed":
        assert res.status is Status.UNSAT and res.nodes == 0
    else:
        assert res.status is Status.UNKNOWN


@pytest.mark.parametrize("driver", _DRIVERS)
def test_point_input_star_is_handled(driver):
    """A star without predicate variables (a single point; its predicate box
    is ``None``) runs through both drivers, and no verdict contradicts the
    point's own output."""
    net = seq_net(2, 5, 5, 2, seed=0)
    x = np.array([0.1, 0.2])
    S = Star(x.reshape(-1, 1), np.zeros((0, 0)), np.zeros((0, 1)))
    with torch.no_grad():
        y0 = float(net(torch.as_tensor(x).unsqueeze(0))[0, 0])
    for tau, truth in ((y0 + 1.0, Status.SAT), (y0 - 1.0, Status.UNSAT)):
        spec = LinearSpec([[1.0, 0.0]], [tau])
        res = _run_driver(driver, S, net, spec)
        assert res.status in (truth, Status.UNKNOWN), (tau, res.status)
        _assert_honest_ce(res, net, spec, x, x)


def test_eval_mode_model_copies_only_training_models():
    m = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(4, 1))
    assert m.training
    e = _eval_mode_model(m)
    assert e is not m
    assert not any(mod.training for mod in e.modules())
    assert m.training is True and m[2].training is True

    m.eval()
    assert _eval_mode_model(m) is m

    def plain(x):
        return x
    assert _eval_mode_model(plain) is plain


def test_verify_refine_checks_counterexamples_in_eval_mode():
    """A training-mode BatchNorm would normalize with batch statistics and
    overwrite the running buffers; the counterexample checks must run the
    eval-mode function on a copy, leaving the caller's model untouched."""
    m = nn.Sequential(nn.Linear(2, 4), nn.Unflatten(1, (1, 2, 2)),
                      nn.BatchNorm2d(1), nn.Flatten(), nn.ReLU(),
                      nn.Linear(4, 1)).double()
    with torch.no_grad():
        m[2].running_mean.fill_(0.3)
        m[2].running_var.fill_(2.0)
    m.train()
    buffers = {k: v.clone() for k, v in m[2].named_buffers()}
    layers = extract_layers(m)      # before the hook: hooked models are rejected
    seen = []
    handle = m.register_forward_pre_hook(
        lambda mod, inp: seen.append(mod.training))
    try:
        _, _, S_in = _box()
        res = verify_refine(S_in, m, LinearSpec([[1.0]], [1e9]),
                            FaithfulSelector(), layers=layers, **CPU_CONFIG)
    finally:
        handle.remove()
    assert res.status is Status.SAT
    assert seen and not any(seen), "a counterexample check ran in train mode"
    assert m.training is True
    for k, v in m[2].named_buffers():
        assert torch.equal(v, buffers[k]), f"BatchNorm buffer {k} changed"


def test_crown_driver_checks_counterexamples_in_eval_mode():
    """Same contract for the CROWN driver (an FC model: ``prepare_layers``
    rejects conv layers), with a training-mode Dropout."""
    m = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(4, 1)).double()
    m.train()
    layers = extract_layers(m)
    seen = []
    handle = m.register_forward_pre_hook(
        lambda mod, inp: seen.append(mod.training))
    try:
        _, _, S_in = _box()
        res = verify_refine_input_crown(S_in, m, LinearSpec([[1.0]], [1e9]),
                                        layers=layers, node_budget=5000)
    finally:
        handle.remove()
    assert res.status is Status.SAT
    assert seen and not any(seen), "a counterexample check ran in train mode"
    assert m.training is True


# --------------------------------------------------------------------------- #
# Low-input-dimension contraction stack on CPU torch                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1])
def test_contract_input_boxes_on_cpu_torch_is_outward_of_exact(
        seed, monkeypatch):
    """Certified domain contraction on CPU torch stays OUTWARD of the exact
    input-dim range over {C alpha <= d, box} and never widens the box."""
    from n2v.refine.tighten import contract_input_boxes

    _gpu_engine_on_cpu_torch(monkeypatch)
    net = seq_net(3, 6, 6, 2, seed=600 + seed)
    S_in = Star.from_bounds(np.array([-1.0, -0.5, -1.0]),
                            np.array([0.5, 1.0, 1.0]))
    root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
    assert root.C.size > 0
    k0 = S_in.nVar

    (clb, cub, empty), = contract_input_boxes(
        [(root.C, root.d)], [(root.predicate_lb, root.predicate_ub)], k0,
        gpu_iters=50, device=None)

    objectives, flags = [], []
    for i in range(k0):
        e = np.zeros(root.nVar)
        e[i] = 1.0
        objectives.extend([e, e])
        flags.extend([True, False])
    exact = root._solve_lp_batch(objectives, flags, LPSolver.DEFAULT)
    assert all(v is not None for v in exact), "test setup: infeasible system"
    assert not empty, "certified-empty on a feasible system"
    for i in range(k0):
        assert clb[i, 0] <= exact[2 * i] + 1e-7, "contraction cut the min"
        assert cub[i, 0] >= exact[2 * i + 1] - 1e-7, "contraction cut the max"
    assert np.all(clb >= root.predicate_lb[:k0] - 1e-12)
    assert np.all(cub <= root.predicate_ub[:k0] + 1e-12)


# (witness_mode, bound_mode) pairs of the contract-split configuration
_CONTRACT_MODES = [("dual_gpu", "dual_gpu"), ("exact", "dual_gpu"),
                   ("dual_gpu", "box")]


@pytest.mark.parametrize("modes", _CONTRACT_MODES,
                         ids=["-".join(m) for m in _CONTRACT_MODES])
@pytest.mark.parametrize("arm", ["inside", "below"])
@pytest.mark.parametrize("seed", [0, 1])
def test_contract_inputs_on_cpu_torch_matches_oracle(seed, arm, modes,
                                                     monkeypatch):
    """The contract-split configuration (layer-ordered splits, certified input
    contraction, wave-batched child reaches, certified GPU prune) on CPU
    torch reproduces the oracle, with honest counterexamples."""
    _gpu_engine_on_cpu_torch(monkeypatch)
    witness_mode, bound_mode = modes
    net, lb, ub, S_in, spec, truth = _tau_instance(seed, arm)
    res = verify_refine(S_in, net, spec, LayerOrderSelector(),
                        bound_mode=bound_mode, prune_mode="dual_gpu",
                        witness_mode=witness_mode, frontier_width=4,
                        contract_inputs=True, gpu_iters=50,
                        node_budget=20000)
    assert res.status == truth, \
        f"seed={seed} {arm} {modes}: {res.status} != {truth}"
    _assert_honest_ce(res, net, spec, lb, ub)
