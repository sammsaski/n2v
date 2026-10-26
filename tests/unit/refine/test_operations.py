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
