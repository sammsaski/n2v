"""
Unit tests for incremental (shared-prefix) reach.

The load-bearing invariant: resuming from a parent checkpoint and reprocessing
only the layers at/after the split must produce a star *identical* to running
``relaxed_reach`` from scratch with the same fixed set. If that holds, incremental
reach is a pure speedup with no behavioural change -- the verdict suite
(test_refine_cpu_sound) then confirms it end-to-end.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine.reach_relaxed import (
    extract_layers, relaxed_reach, resume_reach, resume_reach_pair,
    relu_positions,
)
from n2v.refine.types import NeuronKey, Phase


def _net(seed, dims):
    rng = np.random.default_rng(seed)
    mods = []
    for i in range(len(dims) - 2):
        mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    mods += [nn.Linear(dims[-2], dims[-1])]
    net = nn.Sequential(*mods).double()
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Linear):
                m.weight.copy_(torch.tensor(rng.uniform(-1.4, 1.4, m.weight.shape)))
                m.bias.copy_(torch.tensor(rng.uniform(-0.4, 0.4, m.bias.shape)))
    return net


def _assert_star_equal(a: Star, b: Star, tol=1e-9):
    assert a.V.shape == b.V.shape, f"V shape {a.V.shape} vs {b.V.shape}"
    assert np.allclose(a.V, b.V, atol=tol)
    assert a.C.shape == b.C.shape, f"C shape {a.C.shape} vs {b.C.shape}"
    assert np.allclose(a.C, b.C, atol=tol)
    assert np.allclose(a.d, b.d, atol=tol)
    assert np.allclose(a.predicate_lb, b.predicate_lb, atol=tol)
    assert np.allclose(a.predicate_ub, b.predicate_ub, atol=tol)


def _assert_meta_equal(ma, mb, tol=1e-9):
    assert len(ma) == len(mb), f"meta len {len(ma)} vs {len(mb)}"
    for x, y in zip(ma, mb):
        assert x.key == y.key
        assert x.pred_col == y.pred_col
        assert x.preact_center == pytest.approx(y.preact_center, abs=tol)
        assert np.allclose(x.preact_gens, y.preact_gens, atol=tol)
        assert x.l == pytest.approx(y.l, abs=tol) and x.u == pytest.approx(y.u, abs=tol)


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu"])
@pytest.mark.parametrize("seed", range(4))
def test_resume_equals_from_scratch(bound_mode, seed):
    """resume_reach(parent ckpt, child_fixed) == relaxed_reach(child_fixed)."""
    net = _net(seed, dims=(2, 6, 6, 6, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    parent, pmeta = relaxed_reach(S_in, layers, {}, bound_mode=bound_mode)
    assert parent.checkpoints is not None and len(parent.checkpoints) == len(relu_positions(layers))

    # Fix one relaxed neuron at each layer that has candidates, both phases.
    by_layer = {}
    for m in pmeta:
        by_layer.setdefault(m.key.layer, m.key)
    assert by_layer, "expected relaxed neurons to split on"
    for L, key in by_layer.items():
        for phase in (Phase.ACTIVE, Phase.INACTIVE):
            child_fixed = {key: phase}
            scratch, smeta = relaxed_reach(S_in, layers, child_fixed, bound_mode=bound_mode)
            resumed = resume_reach(parent, layers, child_fixed, bound_mode, L)
            _assert_star_equal(resumed, scratch)
            _assert_meta_equal(resumed.relax_meta, smeta)


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu"])
def test_resume_from_multifix_parent_nonmonotonic(bound_mode):
    """
    The case the BaB actually produces (depth > 1): resume from a parent that
    *already* carries a fix, splitting at a SHALLOWER layer than that fix
    (non-monotonic order). Must still equal a from-scratch reach with the full
    fixed set -- exact V/C/d/meta. This is the coverage the single-fix-from-root
    test misses.
    """
    net = _net(3, dims=(2, 6, 6, 6, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    root, rmeta = relaxed_reach(S_in, layers, {}, bound_mode=bound_mode)

    deep_L = max(m.key.layer for m in rmeta)
    deep_key = next(m.key for m in rmeta if m.key.layer == deep_L)
    # Parent already has a DEEP fix.
    parent = resume_reach(root, layers, {deep_key: Phase.ACTIVE}, bound_mode, deep_L)
    shallow = [m.key for m in parent.relax_meta if m.key.layer < deep_L]
    assert shallow, "need a relaxed neuron shallower than the deep fix"
    shallow_key = shallow[0]

    # Now split the parent at the SHALLOWER layer (non-monotonic).
    combined = {deep_key: Phase.ACTIVE, shallow_key: Phase.INACTIVE}
    child = resume_reach(parent, layers, combined, bound_mode, shallow_key.layer)
    scratch, smeta = relaxed_reach(S_in, layers, combined, bound_mode=bound_mode)
    _assert_star_equal(child, scratch)
    _assert_meta_equal(child.relax_meta, smeta)


def test_resume_rejects_prefix_disagreement():
    """The load-bearing precondition is enforced: a child_fixed that contradicts
    the parent below the split layer raises rather than silently producing an
    unsound star."""
    net = _net(0, dims=(2, 5, 5, 5, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    root, rmeta = relaxed_reach(S_in, layers, {})
    L0_key = next(m.key for m in rmeta if m.key.layer == 0)
    split_L = max(m.key.layer for m in rmeta)
    assert split_L > 0
    # child_fixed adds a fix at layer 0 (< split_L) that the parent lacks.
    with pytest.raises(ValueError, match="below split"):
        resume_reach(root, layers, {L0_key: Phase.ACTIVE}, "box", split_L)
    # out-of-range split layer is also rejected clearly.
    with pytest.raises(ValueError, match="out of range"):
        resume_reach(root, layers, {}, "box", len(root.checkpoints))


def test_checkpoint_prefix_is_shared_by_reference():
    """Child checkpoints/meta below the split layer are the parent's (shared)."""
    net = _net(1, dims=(2, 5, 5, 5, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    parent, pmeta = relaxed_reach(S_in, layers, {})
    L = max(m.key.layer for m in pmeta)  # deepest relaxed layer -> max reuse
    key = next(m.key for m in pmeta if m.key.layer == L)
    child = resume_reach(parent, layers, {key: Phase.ACTIVE}, "box", L)

    # checkpoints strictly below L are the same objects as the parent's.
    for i in range(L):
        assert child.checkpoints[i] is parent.checkpoints[i]
    # meta below L is reused verbatim (same keys/pred_cols as parent's prefix).
    pre_parent = [m for m in pmeta if m.key.layer < L]
    pre_child = [m for m in child.relax_meta if m.key.layer < L]
    _assert_meta_equal(pre_child, pre_parent)


def test_resume_does_less_work_than_scratch():
    """
    Resuming at L>0 actually REUSES the prefix rather than recomputing: the
    child's checkpoints below L are the parent's exact objects (identity), and
    only the n_relu-L layers at/after L are freshly built. Identity (not just
    length) is what proves work was saved, not merely relabelled.
    """
    net = _net(2, dims=(2, 5, 5, 5, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    parent, pmeta = relaxed_reach(S_in, layers, {})
    n_relu = len(relu_positions(layers))
    L = max(m.key.layer for m in pmeta)
    assert L >= 1, "need a non-root split layer for this test"
    key = next(m.key for m in pmeta if m.key.layer == L)
    child = resume_reach(parent, layers, {key: Phase.ACTIVE}, "box", L)
    # prefix [0, L) reused verbatim (same Star objects); suffix [L, n) rebuilt.
    assert all(child.checkpoints[i] is parent.checkpoints[i] for i in range(L))
    assert len(child.checkpoints) == n_relu
    assert child.checkpoints[L] is parent.checkpoints[L]  # entry star at L shared
    # the freshly built suffix is a strict subset of all layers (work saved)
    assert n_relu - L < n_relu


def _parent_with_a_layer0_fix():
    """A parent fixed INACTIVE at a layer-0 neuron, and a layer-1 relaxed
    neuron ``k1`` of that parent."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 6), nn.ReLU(),
                        nn.Linear(6, 1)).double()
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    _, rmeta = relaxed_reach(S_in, layers, {}, bound_mode="box")
    k0 = next(m.key for m in rmeta if m.key.layer == 0)
    parent, _ = relaxed_reach(S_in, layers, {k0: Phase.INACTIVE},
                              bound_mode="box")
    k1 = next(m.key for m in parent.relax_meta if m.key.layer == 1)
    return layers, parent, k1


def test_resume_rejects_a_child_that_drops_a_parent_fix():
    """The shared-prefix check runs in both directions: a parent fix below the
    split layer that the child lacks would silently restrict the child (its
    phase row lives in the reused checkpoint)."""
    layers, parent, k1 = _parent_with_a_layer0_fix()
    with pytest.raises(ValueError, match="below split"):
        resume_reach(parent, layers, {k1: Phase.ACTIVE}, "box", 1)


def test_resume_pair_checks_the_prefix_before_any_gpu_call():
    layers, parent, k1 = _parent_with_a_layer0_fix()
    with pytest.raises(ValueError, match="below split"):
        resume_reach_pair(parent, layers,
                          [{k1: Phase.ACTIVE}, {k1: Phase.INACTIVE}],
                          "dual_gpu", 1)


@pytest.mark.parametrize("call", ["resume_reach", "pair-box", "pair-dual_gpu"])
def test_resume_rejects_a_child_that_flips_a_parent_phase(call):
    """The parent's layer-0 fix is INACTIVE and its phase row lives in the
    reused checkpoint; a child that claims the same neuron ACTIVE below the
    split layer would be built on the wrong region. Catches a prefix check
    that compares only which neurons are fixed, not their phases."""
    layers, parent, k1 = _parent_with_a_layer0_fix()
    k0 = next(k for k in parent.fixed if k.layer == 0)
    flipped = {k0: Phase.ACTIVE, k1: Phase.ACTIVE}
    with pytest.raises(ValueError, match="below split"):
        if call == "resume_reach":
            resume_reach(parent, layers, flipped, "box", 1)
        else:
            resume_reach_pair(parent, layers,
                              [flipped, {k0: Phase.ACTIVE, k1: Phase.INACTIVE}],
                              call.split("-")[1], 1)


@pytest.mark.parametrize("bound_mode", ["box", "dual_gpu"])
def test_resume_pair_checks_the_prefix_of_every_child(bound_mode):
    """Only the SECOND child of the pair disagrees with the parent below the
    split layer; the pair must still refuse."""
    layers, parent, k1 = _parent_with_a_layer0_fix()
    k0 = next(k for k in parent.fixed if k.layer == 0)
    pair = [{k0: Phase.INACTIVE, k1: Phase.ACTIVE},
            {k0: Phase.ACTIVE, k1: Phase.INACTIVE}]
    with pytest.raises(ValueError, match="below split"):
        resume_reach_pair(parent, layers, pair, bound_mode, 1)


@pytest.mark.parametrize("call", ["resume_reach", "pair-dual_gpu"])
def test_resume_rejects_a_parent_without_checkpoints_and_a_split_layer_out_of_range(call):
    """A plain star carries no checkpoints; a split layer of -1 (which Python
    would read as the last ReLU layer) or of the number of ReLU layers names
    no checkpoint."""
    net = _net(0, dims=(2, 5, 5, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    root, _ = relaxed_reach(S_in, layers, {})

    def run(parent, split):
        if call == "resume_reach":
            return resume_reach(parent, layers, {}, "box", split)
        return resume_reach_pair(parent, layers, [{}, {}], "dual_gpu", split)

    with pytest.raises(ValueError, match="no checkpoints"):
        run(S_in, 0)
    for split in (-1, len(root.checkpoints)):
        with pytest.raises(ValueError, match="out of range"):
            run(root, split)


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu"])
def test_grandchildren_resumed_layer_after_layer_equal_from_scratch(bound_mode):
    """A chain in depth order: a child split at layer 0, a grandchild split
    at layer 1 resumed from the child, a great-grandchild at layer 2 resumed
    from the grandchild. Each reads the checkpoints and the recorded bounds
    of a RESUMED star, so a child that stores them shifted by one makes the
    next resume start from another layer's star or reuse another layer's
    bounds; each must equal a reach from scratch with the same fixes."""
    net = _net(5, dims=(2, 6, 6, 6, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    star, _ = relaxed_reach(S_in, layers, {}, bound_mode=bound_mode)
    fixed = {}
    for L in range(len(relu_positions(layers))):
        key = next((m.key for m in star.relax_meta if m.key.layer == L), None)
        if key is None:
            continue
        fixed = {**fixed, key: Phase.ACTIVE if L % 2 else Phase.INACTIVE}
        star = resume_reach(star, layers, fixed, bound_mode, L)
        scratch, smeta = relaxed_reach(S_in, layers, fixed, bound_mode=bound_mode)
        _assert_star_equal(star, scratch)
        _assert_meta_equal(star.relax_meta, smeta)
        assert len(star.checkpoints) == len(relu_positions(layers))
        assert all(np.array_equal(a.V, b.V)
                   for a, b in zip(star.checkpoints, scratch.checkpoints))
    assert len(fixed) >= 2, "setup: the chain split fewer than two layers"
