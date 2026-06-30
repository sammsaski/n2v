"""Unit tests for the refine set-operations (refine / split) and relax_meta."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine import FaithfulSelector, LinearSpec, Status, refine, split, verify_refine
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.witness import make_witness, violation_lp


def _seq(*dims, seed=0):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers).double()


def _problem(seed):
    rng = np.random.default_rng(seed)
    net = _seq(3, 6, 6, 4, seed=seed)
    layers = extract_layers(net)
    lb = rng.uniform(-1, -0.2, size=3)
    ub = rng.uniform(0.2, 1, size=3)
    S_in = Star.from_bounds(lb, ub)
    return net, layers, S_in, lb, ub


def _exact_outputs(net, lb, ub, n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, size=(n, len(lb)))
    with torch.no_grad():
        return net(torch.as_tensor(X, dtype=torch.float64)).numpy()


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
# refine (bound tightening)                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_refine_is_tighter_and_sound(seed):
    net, layers, S_in, lb, ub = _problem(seed)
    S_box, _ = relaxed_reach(S_in, layers, bound_mode="box")
    S_ref = refine(S_box, S_in, layers, "lp_cpu")

    # same fixed-phase region, recorded tighter mode
    assert getattr(S_ref, "fixed") == getattr(S_box, "fixed")
    assert getattr(S_ref, "bound_mode") == "lp_cpu"

    # [[S_ref]] subset of [[S_box]]: its exact LP ranges are tighter-or-equal
    lb_box, ub_box = S_box.get_ranges()
    lb_ref, ub_ref = S_ref.get_ranges()
    assert np.all(lb_ref >= lb_box - 1e-6)
    assert np.all(ub_ref <= ub_box + 1e-6)

    # sound: every exact output lies within the refined star's ranges
    Y = _exact_outputs(net, lb, ub, 400, seed)
    assert np.all(Y >= lb_ref.flatten() - 1e-6)
    assert np.all(Y <= ub_ref.flatten() + 1e-6)


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


def test_split_returns_none_without_metadata():
    _, layers, S_in, _, _ = _problem(0)
    S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))  # no relax_meta
    spec = LinearSpec(np.array([[1.0, 0.0]]), np.array([0.0]))
    res = violation_lp(S, spec, include_Cd=True)
    wit = make_witness(S, spec, res[0], res[1], "faithful")
    assert split(S, S_in, layers, spec, FaithfulSelector(), wit) is None


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
    res = verify_refine(S_in, net, spec, FaithfulSelector(), layers=layers, node_budget=1)
    assert res.status == Status.UNSAT
