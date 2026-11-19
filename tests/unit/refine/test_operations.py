"""Unit tests for the refine set-operations (split, split_wave, phase-pin rows)
and relax_meta."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine import FaithfulSelector, LinearSpec, Status, split, verify_refine
from n2v.refine.operations import _pin_rows
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.types import Phase
from n2v.refine.witness import make_witness, violation_lp
from tests.unit.refine.conftest import seq_net as _seq


CPU_CONFIG = dict(bound_mode="box", prune_mode="exact", witness_mode="exact",
                  frontier_width=1)


def _problem(seed):
    rng = np.random.default_rng(seed)
    net = _seq(3, 6, 6, 4, seed=seed)
    layers = extract_layers(net)
    lb = rng.uniform(-1, -0.2, size=3)
    ub = rng.uniform(0.2, 1, size=3)
    S_in = Star.from_bounds(lb, ub)
    return net, layers, S_in, lb, ub


# --------------------------------------------------------------------------- #
# relax_meta attachment                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_relaxed_reach_attaches_metadata(seed):
    _, layers, S_in, _, _ = _problem(seed)
    S_out, meta = relaxed_reach(S_in, layers, bound_mode="box")
    # the metadata is attached to the star it describes (same object as side-channel)
    assert S_out.relax_meta is meta
    assert len(meta) > 0  # the random net has unstable neurons over this box
    # search provenance is carried too
    assert getattr(S_out, "fixed") == {}
    assert getattr(S_out, "bound_mode") == "box"
    # every meta entry's predicate column is within the star's predicate space
    for nm in meta:
        assert 0 <= nm.pred_col < S_out.nVar


def test_plain_star_has_none_relax_meta():
    S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert S.relax_meta is None


# --------------------------------------------------------------------------- #
# split (witness-guided activation split)                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_returns_two_children_fixing_one_neuron(seed):
    _, layers, S_in, _, _ = _problem(seed)
    S_out, meta = relaxed_reach(S_in, layers, bound_mode="box")
    # a spec whose unsafe region overlaps the output range, so the node is not
    # pruned and a faithful witness exists.
    lb_o, ub_o = S_out.get_ranges()
    G = np.zeros((1, S_out.dim)); G[0, 0] = 1.0          # unsafe := { y0 <= mid }
    g = np.array([0.5 * (lb_o[0, 0] + ub_o[0, 0])])
    spec = LinearSpec(G, g)

    res = violation_lp(S_out, spec, include_Cd=True)
    assert res is not None
    alpha, t = res
    wit = make_witness(S_out, spec, alpha, t, "faithful")

    children = split(S_out, S_in, layers, spec, FaithfulSelector(), wit)
    assert children is not None and len(children) == 2

    parent_fixed = getattr(S_out, "fixed")
    keys = set(meta_nm.key for meta_nm in meta)
    extras = []
    for c in children:
        cf = getattr(c, "fixed")
        assert len(cf) == len(parent_fixed) + 1
        extra = set(cf) - set(parent_fixed)
        assert len(extra) == 1
        k = extra.pop()
        assert k in keys                      # split an actual relaxed neuron
        extras.append((k, cf[k]))

    # both children split the SAME neuron, opposite phases
    (k0, p0), (k1, p1) = extras
    assert k0 == k1
    assert p0 != p1


def test_split_wave_fallback_reach_forwards_gpu_iters(monkeypatch):
    """split_wave's per-child fallback reach (parents not in dual_gpu mode)
    must forward the caller's ``gpu_iters``, like the group reach does."""
    import n2v.refine.operations as ops

    _, layers, S_in, _, _ = _problem(0)
    root, _ = relaxed_reach(S_in, layers, bound_mode="box")
    lb_o, ub_o = root.get_ranges()
    G = np.zeros((1, root.dim)); G[0, 0] = 1.0           # unsafe := { y0 <= mid }
    spec = LinearSpec(G, np.array([0.5 * (lb_o[0, 0] + ub_o[0, 0])]))
    alpha, t = violation_lp(root, spec, include_Cd=True)
    wit = make_witness(root, spec, alpha, t, "faithful")

    # identity contraction: every child keeps its parent's input box
    monkeypatch.setattr(
        "n2v.refine.tighten.contract_input_boxes",
        lambda systems, boxes, k0, **kw: [
            (np.asarray(b[0]).reshape(-1, 1)[:k0],
             np.asarray(b[1]).reshape(-1, 1)[:k0], False) for b in boxes])
    recorded = []
    real_relaxed_reach = ops.relaxed_reach

    def _recording(*args, **kwargs):
        recorded.append(kwargs)
        return real_relaxed_reach(*args, **kwargs)

    monkeypatch.setattr(ops, "relaxed_reach", _recording)
    ops.split_wave([(root, wit)], S_in, layers, spec, FaithfulSelector(),
                   gpu_iters=7)
    assert recorded, "the fallback reach never ran"
    assert all(k.get("gpu_iters") == 7 for k in recorded)


def test_split_returns_none_without_metadata():
    _, layers, S_in, _, _ = _problem(0)
    S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))  # no relax_meta
    spec = LinearSpec(np.array([[1.0, 0.0]]), np.array([0.0]))
    res = violation_lp(S, spec, include_Cd=True)
    wit = make_witness(S, spec, res[0], res[1], "faithful")
    assert split(S, S_in, layers, spec, FaithfulSelector(), wit) is None


# --------------------------------------------------------------------------- #
# phase-pin rows (_pin_rows)                                                   #
# --------------------------------------------------------------------------- #
def _net(seed: int, dims=(2, 4, 4, 2)) -> nn.Module:
    """An FC ReLU net with reproducible weights."""
    rng = np.random.default_rng(seed)
    mods = []
    for i in range(len(dims) - 2):
        mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    mods += [nn.Linear(dims[-2], dims[-1])]
    net = nn.Sequential(*mods).double()
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Linear):
                m.weight.copy_(torch.tensor(rng.uniform(-1.5, 1.5, m.weight.shape)))
                m.bias.copy_(torch.tensor(rng.uniform(-0.5, 0.5, m.bias.shape)))
    return net


def _root(net):
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    S_out, meta = relaxed_reach(S_in, layers)
    return layers, S_in, S_out, meta


def test_pin_rows_match_forced_phase_reach():
    """
    ORACLE: single hidden layer -> the violation LP over the output star with
    the neuron's pin rows appended equals the forced-phase relaxed_reach child
    bound, exactly. Nothing is downstream of the only ReLU layer, so the
    no-re-propagation pin and a full forced re-propagation must agree.
    """
    for seed in range(6):
        net = _net(seed, dims=(2, 5, 2))
        layers, S_in, S_out, meta = _root(net)
        spec = LinearSpec(G=np.array([[1.0, -1.0]]), g=np.array([0.1]))
        for nm in meta:
            for active, phase in ((True, Phase.ACTIVE), (False, Phase.INACTIVE)):
                A, b = _pin_rows(nm, S_out.nVar, active=active)
                pinned_star = Star(S_out.V, np.vstack([S_out.C, A]),
                                   np.vstack([S_out.d, b.reshape(-1, 1)]),
                                   S_out.predicate_lb, S_out.predicate_ub)
                pinned = violation_lp(pinned_star, spec, include_Cd=True)
                child, _ = relaxed_reach(S_in, layers, {nm.key: phase})
                reprop = violation_lp(child, spec, include_Cd=True)
                assert (pinned is None) == (reprop is None)
                if pinned is not None:
                    assert pinned[1] == pytest.approx(reprop[1], abs=1e-6)


# --------------------------------------------------------------------------- #
# driver integration: verdict via verify_specification                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_verify_refine_certifies_disjoint_spec_unsat(seed):
    net, layers, S_in, _, _ = _problem(seed)
    S_out, _ = relaxed_reach(S_in, layers, bound_mode="box")
    lb_o, ub_o = S_out.get_ranges()
    # unsafe region far below the reachable range on dim 0 -> provably disjoint
    G = np.zeros((1, S_out.dim)); G[0, 0] = 1.0
    g = np.array([lb_o[0, 0] - 10.0])
    spec = LinearSpec(G, g)
    res = verify_refine(S_in, net, spec, FaithfulSelector(), layers=layers,
                        node_budget=1, **CPU_CONFIG)
    assert res.status == Status.UNSAT


# --------------------------------------------------------------------------- #
# Witness phase, child order, bound mode,                                      #
# deadline, nothing to split, and the pin rows in exact arithmetic             #
# --------------------------------------------------------------------------- #
# A dyadic 2-3-3-2 net whose first neuron is n0 = x0 - 0.25. The input box
# x0 in [-1.75, 2.25], x1 in [-1, 3] has power-of-two half-widths, so the
# predicate point of x0 = 0.25 is exactly 0 and the witness lies exactly on
# n0's hyperplane.
_ORDER_LO = np.array([-1.75, -1.0])
_ORDER_HI = np.array([2.25, 3.0])
_ORDER_SPEC = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([0.0]))
_WITNESS_X0 = {"inactive": -1.0, "active": 1.0, "hyperplane": 0.25}


def _order_net():
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 3), nn.ReLU(),
                        nn.Linear(3, 2)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.5]]))
        net[0].bias.copy_(torch.tensor([-0.25, 0.5, 4.0]))
        net[2].weight.copy_(torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, -0.5],
                                          [-1.0, 0.0, 0.25]]))
        net[2].bias.copy_(torch.tensor([-0.5, 1.0, -0.5]))
        net[4].weight.copy_(torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, 0.5]]))
        net[4].bias.zero_()
    return net


def _witness_for_input(S, S_in, x):
    """A witness whose input coordinates are those of the input ``x`` (the
    other predicate variables 0)."""
    from n2v.refine.types import Witness

    alpha = np.zeros(S.nVar)
    alpha[:S_in.nVar] = np.linalg.solve(S_in.V[:, 1:], x - S_in.V[:, 0])
    return Witness(alpha, 1.0, 0, "faithful")


def _expected_phase_of_n0(x):
    """The phase of n0 = x0 - 0.25 at ``x``, in exact arithmetic: ACTIVE when
    the pre-activation is >= 0 (the ReLU's identity branch on the
    boundary)."""
    from fractions import Fraction

    return (Phase.ACTIVE if Fraction(float(x[0])) - Fraction(1, 4) >= 0
            else Phase.INACTIVE)


def test_witness_phase_is_active_on_the_hyperplane():
    """``witness_phase`` reads the pre-activation ``c + g . alpha[:len(g)]``
    at the witness: ACTIVE above 0 and exactly on 0, INACTIVE below 0, and
    the predicate variables after the neuron's prefix do not count. Catches
    a strict comparison (a witness on the hyperplane claims INACTIVE) and a
    read of the wrong part of ``alpha``."""
    from n2v.refine import witness_phase
    from n2v.refine.types import NeuronKey, NeuronMeta

    nm = NeuronMeta(key=NeuronKey(0, 1), pred_col=2, preact_center=-0.75,
                    preact_gens=np.array([0.5, 0.25]), l=-2.0, u=1.0)
    cases = [([1.0, 1.0, -7.0], Phase.ACTIVE),        # exactly 0
             ([1.0, 1.5, -7.0], Phase.ACTIVE),        # 0.125
             ([1.0, 0.5, 9.0], Phase.INACTIVE),       # -0.125
             ([1.0, 3.0, 9.0], Phase.ACTIVE),         # 0.5
             ([2.0, -1.0, -100.0], Phase.ACTIVE),     # exactly 0
             ([2.0, -2.0, 100.0], Phase.INACTIVE)]    # -0.25
    for alpha, phase in cases:
        assert witness_phase(np.array(alpha), nm) == phase, alpha


@pytest.mark.parametrize("side", list(_WITNESS_X0))
@pytest.mark.parametrize("bound_mode, incremental", [
    ("box", True), ("box", False), ("lp_cpu", True), ("dual_gpu", True),
    ("dual_gpu", False)])
def test_split_children_are_witness_phase_first(bound_mode, incremental, side,
                                                engine_device):
    """``split`` returns two children that fix the chosen neuron to opposite
    phases, the phase the witness claims first (element 0), on every path
    (one resume per child, the lockstep pair under dual_gpu, and the
    re-reach from the input), with the witness below, above and exactly on
    the neuron's hyperplane (where the claim is ACTIVE). The expected phase
    is the sign of n0 = x0 - 0.25 at the witness input, in fractions.
    Catches children built in the other order on any path, a strict
    comparison on the hyperplane, and children re-reached with the parent's
    phases instead of their own."""
    from n2v.refine import LayerOrderSelector
    from n2v.refine.types import NeuronKey

    net = _order_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_ORDER_LO, _ORDER_HI)
    root, _ = relaxed_reach(S_in, layers, bound_mode=bound_mode)
    x = np.array([_WITNESS_X0[side], 1.0])
    children = split(root, S_in, layers, _ORDER_SPEC, LayerOrderSelector(),
                     _witness_for_input(root, S_in, x),
                     incremental=incremental)
    key = NeuronKey(0, 0)
    first = _expected_phase_of_n0(x)
    other = Phase.INACTIVE if first == Phase.ACTIVE else Phase.ACTIVE
    assert children is not None and len(children) == 2
    assert [c.fixed.get(key) for c in children] == [first, other]
    assert all(c.bound_mode == bound_mode for c in children)


@pytest.mark.parametrize("side", list(_WITNESS_X0))
@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu", "dual_gpu"])
def test_split_wave_children_are_witness_phase_first(bound_mode, side,
                                                     engine_device):
    """``split_wave`` returns, per job, the children of a split on the
    chosen neuron in witness-phase-first order, each reached at its parent's
    bound mode (the group reach for a dual_gpu wave, one reach per child at
    the parent's own mode otherwise) and fixing exactly the split neuron.
    Both phase regions of n0 hold inputs, so no child may be dropped.
    Catches children in the other order, an lp_cpu wave reached as a
    dual_gpu one, and a child that carries another child's fixed phases."""
    from n2v.refine import LayerOrderSelector, split_wave
    from n2v.refine.types import NeuronKey

    net = _order_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_ORDER_LO, _ORDER_HI)
    root, _ = relaxed_reach(S_in, layers, bound_mode=bound_mode)
    x = np.array([_WITNESS_X0[side], 1.0])
    res = split_wave([(root, _witness_for_input(root, S_in, x))], S_in,
                     layers, _ORDER_SPEC, LayerOrderSelector())
    key = NeuronKey(0, 0)
    first = _expected_phase_of_n0(x)
    other = Phase.INACTIVE if first == Phase.ACTIVE else Phase.ACTIVE
    assert len(res) == 1 and res[0] is not None and len(res[0]) == 2
    assert [c.fixed for c in res[0]] == [{key: first}, {key: other}]
    assert all(c.bound_mode == bound_mode for c in res[0])


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu", "dual_gpu"])
def test_split_wave_after_its_deadline_leaves_every_job_undecided(bound_mode,
                                                                  engine_device):
    """``split_wave`` with a deadline already in the past reaches no child
    and leaves every job ``None`` (the caller's UNKNOWN), on the per-child
    path (box, lp_cpu) and on the group reach (dual_gpu). A job that comes
    back as an empty list would claim its parent region is empty, a prune;
    children returned after the deadline ignore it. Catches both."""
    import time

    from n2v.refine import LayerOrderSelector, split_wave
    from n2v.refine.types import NeuronKey

    net = _order_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_ORDER_LO, _ORDER_HI)
    stars = [relaxed_reach(S_in, layers, f, bound_mode=bound_mode)[0]
             for f in ({}, {NeuronKey(0, 0): Phase.INACTIVE})]
    jobs = [(S, _witness_for_input(S, S_in, np.array([-1.0, 1.0])))
            for S in stars]
    res = split_wave(jobs, S_in, layers, _ORDER_SPEC, LayerOrderSelector(),
                     deadline=time.perf_counter() - 1.0)
    assert res == [None, None]


class _NoChoice(FaithfulSelector):
    """A selector that returns no key."""

    def choose(self, *args, **kwargs):
        return None


class _AbsentKey(FaithfulSelector):
    """A selector that returns a key the star has no metadata for."""

    def choose(self, *args, **kwargs):
        from n2v.refine.types import NeuronKey

        return NeuronKey(7, 99)


@pytest.mark.parametrize("selector", [_NoChoice(), _AbsentKey()],
                         ids=["no-key", "absent-key"])
def test_nothing_to_split_returns_none(selector):
    """``split`` returns ``None`` and ``split_wave`` leaves the job ``None``
    when the selector gives no key or a key absent from the star's
    metadata: the caller then keeps the branch undecided. An empty list
    instead would be read as two certified-empty children, a prune of a
    region that holds inputs. An empty wave returns an empty list."""
    from n2v.refine import split_wave

    net = _order_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_ORDER_LO, _ORDER_HI)
    root, meta = relaxed_reach(S_in, layers, bound_mode="box")
    assert meta, "test setup: the root has relaxed neurons"
    wit = _witness_for_input(root, S_in, np.array([-1.0, 1.0]))
    for incremental in (True, False):
        assert split(root, S_in, layers, _ORDER_SPEC, selector, wit,
                     incremental=incremental) is None
    assert split_wave([(root, wit)], S_in, layers, _ORDER_SPEC,
                      selector) == [None]
    assert split_wave([], S_in, layers, _ORDER_SPEC, selector) == []


def test_pin_rows_state_each_phase_exactly():
    """``_pin_rows(nm, nVar, active)`` returns rows ``A a <= b`` that hold,
    in exact arithmetic, exactly when the neuron's pre-activation
    ``p = c + g . a[:len(g)]`` has the phase and the neuron's own variable
    ``a[pred_col]`` equals its ReLU value there: ``p >= 0`` and
    ``a[pred_col] == p`` (ACTIVE), ``p <= 0`` and ``a[pred_col] == 0``
    (INACTIVE). The neuron's variable is not the last one and the
    generators cover a prefix of the variables, so a pin on the wrong
    variable, a missing padding and a right-hand side of the wrong sign
    each change the answer at some point. Points on both sides of the
    hyperplane, on it, and with the variable equal or not equal to the
    value are checked."""
    from fractions import Fraction
    from itertools import product

    from n2v.refine.types import NeuronKey, NeuronMeta

    nm = NeuronMeta(key=NeuronKey(1, 2), pred_col=3, preact_center=-0.375,
                    preact_gens=np.array([0.5, -1.25, 0.75]), l=-3.0, u=2.0)
    nVar = 5
    F = Fraction
    checked = {True: [0, 0], False: [0, 0]}
    for active in (True, False):
        A, b = _pin_rows(nm, nVar, active=active)
        A = [[F(float(v)) for v in row] for row in np.atleast_2d(A)]
        b = [F(float(v)) for v in np.asarray(b).reshape(-1)]
        assert all(len(row) == nVar for row in A)
        for a0, a1, a2, a4 in product((-1.5, 0.25, 2.0), (-0.5, 0.75),
                                      (-1.0, 0.5), (-2.0, 3.0)):
            pre = (F(-3, 8) + F(1, 2) * F(a0) - F(5, 4) * F(a1)
                   + F(3, 4) * F(a2))
            for r in (pre, F(0), pre + F(1, 8), -pre):
                a = [F(a0), F(a1), F(a2), r, F(a4)]
                holds = all(sum(x * y for x, y in zip(row, a)) <= rhs
                            for row, rhs in zip(A, b))
                if active:
                    want = pre >= 0 and r == pre
                else:
                    want = pre <= 0 and r == 0
                assert holds == want, (active, a, float(pre))
                checked[active][int(want)] += 1
    assert all(n > 0 for v in checked.values() for n in v), checked


@pytest.mark.parametrize("bound_mode, incremental, callee", [
    ("box", True, "resume_reach"), ("dual_gpu", True, "resume_reach_pair"),
    ("box", False, "relaxed_reach")])
def test_split_hands_on_its_options(bound_mode, incremental, callee,
                                    monkeypatch, engine_device):
    """``split`` hands ``gpu_iters`` to the child reach on every path and
    ``warm_starts`` to the incremental reaches (one resume per child, the
    lockstep pair under dual_gpu); ``split_wave`` hands ``gpu_iters`` to the
    group reach of a dual_gpu wave, whose population bounds then use it.
    Catches an option that is dropped on one path (the reach then runs at
    its default budget or with warm starts the caller turned off)."""
    import n2v.refine.operations as ops

    net = _order_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_ORDER_LO, _ORDER_HI)
    root, _ = relaxed_reach(S_in, layers, bound_mode=bound_mode)
    wit = _witness_for_input(root, S_in, np.array([-1.0, 1.0]))
    recorded = []
    real = getattr(ops, callee)

    def _recording(*args, **kwargs):
        recorded.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ops, callee, _recording)
    from n2v.refine import LayerOrderSelector

    children = ops.split(root, S_in, layers, _ORDER_SPEC, LayerOrderSelector(),
                         wit, incremental=incremental, gpu_iters=7,
                         warm_starts=False)
    assert children is not None and recorded
    assert all(k.get("gpu_iters") == 7 for k in recorded), recorded
    if callee != "relaxed_reach":
        assert all(k.get("warm_starts") is False for k in recorded), recorded


def test_split_wave_hands_gpu_iters_to_the_group_reach(monkeypatch,
                                                       engine_device):
    """A dual_gpu wave reaches its children by the group reach, whose
    per-layer population bounds must use the caller's ``gpu_iters``.
    Catches the option dropped between ``split_wave`` and the group
    reach."""
    import n2v.refine.tighten as tighten
    from n2v.refine import LayerOrderSelector, split_wave

    net = _order_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_ORDER_LO, _ORDER_HI)
    root, _ = relaxed_reach(S_in, layers, bound_mode="dual_gpu")
    wit = _witness_for_input(root, S_in, np.array([-1.0, 1.0]))
    recorded = []
    real = tighten.neuron_bounds_population

    def _recording(*args, **kwargs):
        recorded.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(tighten, "neuron_bounds_population", _recording)
    res = split_wave([(root, wit)], S_in, layers, _ORDER_SPEC,
                     LayerOrderSelector(), gpu_iters=7)
    assert res[0] is not None and recorded
    assert all(k.get("gpu_iters") == 7 for k in recorded), recorded


@pytest.mark.parametrize("bound_mode", ["box", "dual_gpu"])
def test_split_wave_refuses_an_unknown_layer(bound_mode, engine_device):
    """A layer list holding an object that is neither an affine layer nor a
    ReLU layer is refused with ``TypeError`` ("Unknown layer type") by both
    child reaches of ``split_wave`` (one reach per child, and the group
    reach of a dual_gpu wave), instead of skipping the layer and returning
    children of a different network."""
    from n2v.refine import LayerOrderSelector, split_wave

    net = _order_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_ORDER_LO, _ORDER_HI)
    root, _ = relaxed_reach(S_in, layers, bound_mode=bound_mode)
    wit = _witness_for_input(root, S_in, np.array([-1.0, 1.0]))
    with pytest.raises(TypeError, match="Unknown layer type"):
        split_wave([(root, wit)], S_in, layers + [object()], _ORDER_SPEC,
                   LayerOrderSelector())
