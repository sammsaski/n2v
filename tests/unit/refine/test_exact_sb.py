"""
Unit tests for ExactStarSBSelector and the exact strong-branching score.

The central claim under test is the proposal's structural theorem: the exact
post-split child bound is one LP on the *existing* output star with the neuron
un-relaxed by the equality-pin discipline (``alpha_r = g_i`` active / ``= 0``
inactive) -- no re-propagation. We pin three facts:

  1. ORACLE: on a single-hidden-layer net (nothing downstream to re-tighten) the
     pinned child bound ``m_i^pm`` equals ``violation_lp`` of the forced-phase
     ``relaxed_reach`` child *exactly* -- an independent code path. This validates
     that the pin encodes the true forced phase.
  2. DISCRIMINATION (regression guard): the equality pin makes both children
     strictly improve where the witness had relaxation slack, so the score
     actually ranks neurons. The degenerate halfspace-only cut (no pin) instead
     gives ``min(child margins) == t_parent`` for *every* candidate -- the bug
     this test exists to catch.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.selectors import ExactStarSBSelector, _pin_rows
from n2v.refine.types import LinearSpec, Phase
from n2v.refine.witness import (
    augmented_violation_lp,
    epsilon_vector,
    make_witness,
    violation_lp,
)


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


def _halfspace_only_rows(nm, nVar, active):
    """The OLD (buggy) cut: pre-activation halfspace only, no equality pin."""
    g = np.zeros(nVar)
    g[: len(nm.preact_gens)] = nm.preact_gens
    if active:   # g_i >= 0
        return (-g)[None, :], np.array([nm.preact_center])
    return g[None, :], np.array([-nm.preact_center])  # g_i <= 0


def test_pin_matches_forced_phase_reach():
    """
    ORACLE: single hidden layer -> pinned m_i^pm == forced-phase relaxed_reach
    child bound, exactly. Nothing is downstream of the only ReLU layer, so the
    no-re-propagation pin and a full forced re-propagation must agree.
    """
    for seed in range(6):
        net = _net(seed, dims=(2, 5, 2))
        layers, S_in, S_out, meta = _root(net)
        spec = LinearSpec(G=np.array([[1.0, -1.0]]), g=np.array([0.1]))
        for nm in meta:
            for active, phase in ((True, Phase.ACTIVE), (False, Phase.INACTIVE)):
                A, b = _pin_rows(nm, S_out.nVar, active=active)
                pinned = augmented_violation_lp(S_out, spec, A, b, include_Cd=True)
                child, _ = relaxed_reach(S_in, layers, {nm.key: phase})
                reprop = violation_lp(child, spec, include_Cd=True)
                assert (pinned is None) == (reprop is None)
                if pinned is not None:
                    assert pinned[1] == pytest.approx(reprop[1], abs=1e-6)


def test_pin_discriminates_where_halfspace_only_is_degenerate():
    """
    REGRESSION GUARD for the degenerate-default bug. With the equality pin, a
    neuron whose witness carries relaxation slack has BOTH children strictly
    improved, so the score ranks it above t_parent. The halfspace-only cut (no
    pin) gives min(child margins) == t_parent for EVERY candidate (the witness
    survives into one child) -- a no-op selector. We assert both, on a node
    chosen to have slack.
    """
    found = False
    for seed in range(40):
        net = _net(seed, dims=(2, 6, 6, 2))
        _, _, S_out, meta = _root(net)
        if not meta:
            continue
        # Unsafe region covering the relaxed set so the root is feasible (t<=0).
        lo, hi = S_out.get_range(0)
        spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([float(hi)]))
        parent = violation_lp(S_out, spec, include_Cd=True)
        if parent is None or parent[1] > -1e-6:
            continue
        alpha, t_parent = parent
        eps = epsilon_vector(alpha, meta)
        if eps.max(initial=0.0) < 1e-3:
            continue  # need a witness with relaxation slack

        any_pinned_improves = False
        for nm in meta:
            # halfspace-only (buggy) cut: one child always recovers t_parent.
            hs = []
            for active in (True, False):
                A, b = _halfspace_only_rows(nm, S_out.nVar, active)
                r = augmented_violation_lp(S_out, spec, A, b, include_Cd=True)
                hs.append(t_parent if r is None else r[1])
            assert min(hs) == pytest.approx(t_parent, abs=1e-6), (
                "halfspace-only cut should be degenerate (partition theorem)"
            )
            # pinned cut: both children can strictly improve.
            pin = []
            for active in (True, False):
                A, b = _pin_rows(nm, S_out.nVar, active=active)
                r = augmented_violation_lp(S_out, spec, A, b, include_Cd=True)
                pin.append(np.inf if r is None else r[1])
            assert min(pin) >= t_parent - 1e-6
            if min(pin) > t_parent + 1e-6:
                any_pinned_improves = True
        if any_pinned_improves:
            found = True
            break
    assert found, "no node exercised the pin's discrimination over halfspace-only"


def test_selector_picks_a_real_candidate():
    """choose() returns a key present in meta (and None on empty meta)."""
    net = _net(2)
    _, _, S_out, meta = _root(net)
    spec = LinearSpec(G=np.array([[1.0, -1.0]]), g=np.array([0.2]))
    wit_alpha, wit_t = violation_lp(S_out, spec, include_Cd=True)
    wit = make_witness(S_out, spec, wit_alpha, wit_t, "faithful")

    sel = ExactStarSBSelector()
    key = sel.choose(S_out, spec, meta, wit)
    assert key in {nm.key for nm in meta}
    assert sel.choose(S_out, spec, [], wit) is None


def test_topk_filter_restricts_to_heuristic_topk():
    """
    top_k=1 exact-scores only the cheap heuristic's #1 pick, so the choice equals
    FaithfulSelector's. With enough candidates, the unfiltered (top_k=None) form
    can pick a different neuron -- evidence the exact look-ahead overrides the
    heuristic when it scores more candidates.
    """
    from n2v.refine.selectors import FaithfulSelector
    net = _net(7, dims=(2, 8, 8, 2))
    _, _, S_out, meta = _root(net)
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([float(S_out.get_range(0)[1])]))
    wa, wt = violation_lp(S_out, spec, include_Cd=True)
    wit = make_witness(S_out, spec, wa, wt, "faithful")
    assert len(meta) > 1

    k1 = ExactStarSBSelector(top_k=1).choose(S_out, spec, meta, wit)
    assert k1 == FaithfulSelector().choose(S_out, spec, meta, wit)

    # top_k >= |meta| is identical to scoring all (no filtering applied).
    big = ExactStarSBSelector(top_k=len(meta)).choose(S_out, spec, meta, wit)
    full = ExactStarSBSelector(top_k=None).choose(S_out, spec, meta, wit)
    assert big == full


def test_topk_validation():
    with pytest.raises(ValueError):
        ExactStarSBSelector(top_k=0)
    with pytest.raises(ValueError):
        ExactStarSBSelector(top_k=-3)


def test_combiner_validation_and_dispatch():
    """Only the lean {min, sum} surface is valid; both select a real candidate."""
    with pytest.raises(ValueError):
        ExactStarSBSelector(combiner="nope")
    for dead in ("product", "max"):  # dropped: degenerate / NaN-prone
        with pytest.raises(ValueError):
            ExactStarSBSelector(combiner=dead)
    net = _net(3)
    _, _, S_out, meta = _root(net)
    spec = LinearSpec(G=np.array([[1.0, -1.0]]), g=np.array([0.2]))
    wit_alpha, wit_t = violation_lp(S_out, spec, include_Cd=True)
    wit = make_witness(S_out, spec, wit_alpha, wit_t, "faithful")
    for combiner in ("min", "sum"):
        key = ExactStarSBSelector(combiner=combiner).choose(S_out, spec, meta, wit)
        assert key in {nm.key for nm in meta}
